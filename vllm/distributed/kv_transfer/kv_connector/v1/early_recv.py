# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Report finished KV loads to the scheduler ahead of the step outputs.

Workers report each load as soon as they see it finish. Once every worker has
reported a load, the scheduler can resume the request in its next schedule().
The step outputs still report every load; the scheduler-side connector drops
the loads it already reported, so a lost report only delays the request.
"""

import threading
from collections.abc import Iterable

import msgspec
import zmq

from vllm.utils.network_utils import get_open_zmq_ipc_path, make_zmq_socket
from vllm.v1.outputs import KVConnectorOutput

# Reports queued for a scheduler that does not read them (e.g. one on another
# node) are dropped beyond this many.
_MAX_QUEUED_REPORTS = 1024


class EarlyRecvReporter:
    """Worker side: sends the loads this worker finished to the scheduler."""

    def __init__(self, rank: int) -> None:
        self._rank = rank
        self._path: str | None = None
        self._socket: zmq.Socket | None = None
        # Connector threads may report while the worker thread connects.
        self._lock = threading.Lock()

    def connect(self, path: str | None) -> None:
        """Connect to the scheduler's address from the connector metadata."""
        if path is None or path == self._path:
            return
        with self._lock:
            if self._socket is not None:
                self._socket.close(linger=0)
            self._socket = zmq.Context.instance().socket(zmq.PUSH)
            self._socket.setsockopt(zmq.SNDHWM, _MAX_QUEUED_REPORTS)
            self._socket.setsockopt(zmq.LINGER, 0)
            self._socket.connect(path)
            self._path = path

    def report(self, req_ids: Iterable[str]) -> None:
        req_ids = list(req_ids)
        if not req_ids:
            return
        message = msgspec.msgpack.encode((self._rank, req_ids))
        with self._lock:
            if self._socket is None:
                return
            try:
                self._socket.send(message, zmq.NOBLOCK)
            except zmq.Again:
                # The step outputs still report these loads.
                pass

    def close(self) -> None:
        with self._lock:
            if self._socket is not None:
                self._socket.close(linger=0)
                self._socket = None


class EarlyRecvTracker:
    """Scheduler side: counts the workers' reports of each pending load."""

    def __init__(self, num_workers: int) -> None:
        self.path = get_open_zmq_ipc_path()
        self._socket = make_zmq_socket(zmq.Context.instance(), self.path, zmq.PULL)
        self._decoder = msgspec.msgpack.Decoder(tuple[int, list[str]])
        self._num_workers = num_workers
        # Pending loads and the workers that reported them finished.
        self._pending: dict[str, set[int]] = {}
        # Loads reported finished here whose step outputs have not arrived.
        self._finished: set[str] = set()

    def add_load(self, req_id: str) -> None:
        self._pending[req_id] = set()

    def awaits_step_output(self, req_id: str) -> bool:
        """Whether a load of the request was reported here and its step
        outputs have not arrived yet."""
        return req_id in self._finished

    def take_finished(self) -> set[str]:
        """Loads that every worker reported since the last call."""
        finished = set[str]()
        while True:
            try:
                message = self._socket.recv(zmq.NOBLOCK)
            except zmq.Again:
                break
            rank, req_ids = self._decoder.decode(message)
            for req_id in req_ids:
                ranks = self._pending.get(req_id)
                if ranks is None:
                    continue
                ranks.add(rank)
                if len(ranks) == self._num_workers:
                    del self._pending[req_id]
                    finished.add(req_id)
        self._finished |= finished
        return finished

    def update_connector_output(self, output: KVConnectorOutput) -> None:
        """Drop the loads already reported here from the step outputs."""
        if not output.finished_recving:
            return
        reported = output.finished_recving & self._finished
        self._finished -= reported
        output.finished_recving -= reported
        for req_id in output.finished_recving:
            self._pending.pop(req_id, None)

    def request_finished(self, req_id: str) -> None:
        """Stop counting reports of a load whose request finished; its step
        outputs then free its blocks."""
        self._pending.pop(req_id, None)

    def close(self) -> None:
        self._socket.close(linger=0)
