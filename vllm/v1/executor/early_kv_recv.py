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
from vllm.distributed.utils import sched_yield
from vllm.logger import init_logger
from vllm.utils.network_utils import get_open_zmq_ipc_path, make_zmq_socket
from vllm.v1.outputs import KVConnectorOutput

logger = init_logger(__name__)

# Interval between polls while the worker waits for its next RPC.
_POLL_INTERVAL_S = 0.0002
# Without a step RPC for this long the engine is idle or paused: stop polling
# and let the worker block in dequeue().
_ACTIVE_WINDOW_S = 1.0


class EarlyKVOutputAggregator(KVOutputAggregator):
    """Also counts receives that workers report between step outputs."""

    def add_finished_recving(self, req_ids: set[str]) -> set[str]:
        """Count one worker's report and return the requests now finished.

        Each worker reports a load once, so votes are not deduplicated.
        """
        failed = req_ids & self._failed_recving_pending
        if failed:
            # Let normal aggregation count these votes and deliver the failure.
            self.ft_defer_kv_connector_outputs(
                [KVConnectorOutput(finished_recving=failed)]
            )
        finished = set[str]()
        expected = self._expected_finished_count
        for req_id in req_ids - failed:
            remaining = self._recv_remaining_count.get(req_id, expected) - 1
            if remaining:
                self._recv_remaining_count[req_id] = remaining
            else:
                finished.add(req_id)
                self._recv_remaining_count.pop(req_id, None)
        return finished


class EarlyKVRecvReceiver:
    """Engine end: receives the workers' reports."""

    def __init__(self) -> None:
        self.path = get_open_zmq_ipc_path()
        self._socket = make_zmq_socket(zmq.Context.instance(), self.path, zmq.PULL)

    def drain(self) -> list[set[str]]:
        """Pending reports, one per worker poll; each is a separate vote."""
        notifications: list[set[str]] = []
        while True:
            try:
                notifications.append(self._socket.recv_pyobj(zmq.NOBLOCK))
            except zmq.Again:
                return notifications

    def close(self) -> None:
        self._socket.close(linger=0)


class EarlyKVRecvSender:
    """Worker end: polls the KV connector between RPCs."""

    def __init__(self, path: str) -> None:
        self._socket = make_zmq_socket(zmq.Context.instance(), path, zmq.PUSH)
        self._last_step_rpc = time.perf_counter()
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
        self._last_step_rpc = time.perf_counter()

    def wait_for_poll(self, rpc_mq: MessageQueue) -> bool:
        """Spin until the next poll is due, yielding as dequeue() does.

        Returns False once an RPC is queued, or at once when no step RPC
        arrived recently or no receive is pending.
        """
        if self._trace_no_spin:
            return False
        if (
            time.perf_counter() - self._last_step_rpc > _ACTIVE_WINDOW_S
            or not has_kv_transfer_group()
            or not get_kv_transfer_group().has_pending_recving()
        ):
            return False
        start = time.perf_counter()
        due = False
        while not (rpc_mq.shutting_down or rpc_mq.readable()):
            if time.perf_counter() >= self._next_poll:
                due = True
                break
            sched_yield()
        self._trace_wait_s += time.perf_counter() - start
        self._trace_waits += 1
        if not due:
            self._trace_rpc_after_wait += 1
        return due

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
