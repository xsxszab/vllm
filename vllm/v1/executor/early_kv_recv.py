# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Report completed KV pulls between model outputs and the next schedule()."""

import os
import time

import torch
import zmq

from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.logger import init_logger
from vllm.utils.network_utils import get_open_zmq_ipc_path, make_zmq_socket
from vllm.v1.outputs import KVConnectorOutput

logger = init_logger(__name__)


class EarlyKVOutputAggregator(KVOutputAggregator):
    """Share the normal receive quorum only when early reporting is enabled."""

    def add_finished_recving(self, req_ids: set[str]) -> set[str]:
        """Callers report each worker/load once; identities are not deduplicated."""
        failed = req_ids & self._failed_recving_pending
        if failed:
            # Let normal aggregation count these votes and deliver the failure.
            self.ft_defer_kv_connector_outputs(
                [KVConnectorOutput(finished_recving=failed)]
            )
        finished = set[str]()
        for req_id in req_ids - failed:
            remaining = (
                self._recv_remaining_count.get(req_id, self._expected_finished_count)
                - 1
            )
            if remaining:
                self._recv_remaining_count[req_id] = remaining
            else:
                finished.add(req_id)
                self._recv_remaining_count.pop(req_id, None)
        return finished


class EarlyKVRecvReceiver:
    """Engine core end: the request ids whose KV loads finished."""

    def __init__(self) -> None:
        self.path = get_open_zmq_ipc_path()
        self._socket = make_zmq_socket(zmq.Context.instance(), self.path, zmq.PULL)

    def drain(self) -> list[set[str]]:
        notifications: list[set[str]] = []
        while True:
            try:
                notifications.append(self._socket.recv_pyobj(zmq.NOBLOCK))
            except zmq.Again:
                return notifications

    def close(self) -> None:
        self._socket.close(linger=0)


# Interval between polls while the worker waits for its next RPC.
_POLL_INTERVAL_S = 0.0002
# Without a step RPC for this long the engine is idle or paused: stop polling
# and let the worker block in dequeue().
_ACTIVE_WINDOW_S = 1.0


class EarlyKVRecvSender:
    """Worker end: polls the KV connector between RPCs."""

    def __init__(self, path: str) -> None:
        self._socket = make_zmq_socket(zmq.Context.instance(), path, zmq.PUSH)
        self._last_activity = time.perf_counter()
        self._next_poll = 0.0
        # EXPERIMENT ONLY (not for upstream): disable the between-RPC wait, and
        # count the waits and the RPCs that arrived during one (an upper bound
        # of the pickup delay the wait adds).
        self._trace_no_spin = os.environ.get("VLLM_EARLY_RECV_NO_SPIN") == "1"
        self._trace = os.environ.get("VLLM_EARLY_RECV_TRACE") == "1"
        self._trace_wait_s = 0.0
        self._trace_waits = 0
        self._trace_polls = 0
        self._trace_rpc_after_wait = 0
        self._trace_rpc_after_wait_s = 0.0
        self._trace_next_report = time.perf_counter() + 10.0

    def _trace_report(self, now: float) -> None:
        if not self._trace or now < self._trace_next_report:
            return
        logger.info(
            "EARLY_RECV_TRACE spin_stats - %.6f pid=%d spin_s=%.3f iters=%d "
            "polls=%d rpc_after_wait=%d rpc_after_wait_s=%.4f",
            time.time(),
            os.getpid(),
            self._trace_wait_s,
            self._trace_waits,
            self._trace_polls,
            self._trace_rpc_after_wait,
            self._trace_rpc_after_wait_s,
        )
        self._trace_wait_s = 0.0
        self._trace_waits = 0
        self._trace_polls = 0
        self._trace_rpc_after_wait = 0
        self._trace_rpc_after_wait_s = 0.0
        self._trace_next_report = now + 10.0

    def step_rpc_done(self) -> None:
        """Worker thread: an ``execute_model`` or ``sample_tokens`` RPC finished."""
        self._last_activity = time.perf_counter()

    def wait_for_poll(self, rpc_mq: MessageQueue) -> bool:
        """Wait briefly for a pending receive, yielding to queued RPCs."""
        if self._trace_no_spin:
            return False
        if not has_kv_transfer_group() or rpc_mq.shutting_down or rpc_mq.readable():
            return False
        now = time.perf_counter()
        if (
            now - self._last_activity > _ACTIVE_WINDOW_S
            or not get_kv_transfer_group().has_pending_recving()
        ):
            return False
        if (delay := self._next_poll - now) > 0:
            time.sleep(delay)
            slept = time.perf_counter() - now
            self._trace_wait_s += slept
            self._trace_waits += 1
            if rpc_mq.readable():
                self._trace_rpc_after_wait += 1
                self._trace_rpc_after_wait_s += slept
        # An RPC may have arrived during the bounded wait. Leave it for dequeue.
        return not rpc_mq.shutting_down and not rpc_mq.readable()

    def poll(self) -> None:
        """Send completed receives; let the worker monitor handle fatal errors."""
        self._trace_polls += 1
        self._trace_report(time.perf_counter())
        try:
            with torch.inference_mode():
                req_ids = get_kv_transfer_group().poll_finished_recving()
            if req_ids:
                self._socket.send_pyobj(req_ids)
        except Exception:
            logger.exception("Early KV receive poll failed")
            # Wake the monitor even while the engine is paused. Receive state
            # may already be lost; neither a later RPC nor a retry can recover it.
            os._exit(1)
        finally:
            # Measured from the end of a poll, so long polls do not run back
            # to back.
            self._next_poll = time.perf_counter() + _POLL_INTERVAL_S

    def close(self) -> None:
        self._socket.close(linger=0)
