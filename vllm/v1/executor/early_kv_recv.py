# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine side of early KV receive reporting: collect the workers' reports and
count them in the receive quorum before the next schedule()."""

from itertools import chain

import zmq

from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.utils.network_utils import get_open_zmq_ipc_path, make_zmq_socket
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput


class EarlyKVOutputAggregator(KVOutputAggregator):
    """Also counts receives that workers report between step outputs."""

    def __init__(self, expected_finished_count: int) -> None:
        super().__init__(expected_finished_count)
        # Unfinished receives with a vote from a step output.
        self._step_output_recving = set[str]()

    def aggregate(
        self, outputs: list[ModelRunnerOutput | None], output_rank: int = 0
    ) -> ModelRunnerOutput | None:
        voted = {
            req_id
            for kv_output in chain(
                self._salvaged_outputs,
                (o.kv_connector_output for o in outputs if o is not None),
            )
            if kv_output is not None
            for req_id in kv_output.finished_recving or ()
        }
        output = super().aggregate(outputs, output_rank)
        if output is not None:
            self._step_output_recving = {
                req_id
                for req_id in self._step_output_recving | voted
                if req_id in self._recv_remaining_count
            }
        return output

    def add_finished_recving(self, req_ids: set[str]) -> set[str]:
        """Count one worker's report and return the requests now finished.

        Each worker reports a load once, so votes are not deduplicated.
        """
        deferred = req_ids & (self._failed_recving_pending | self._step_output_recving)
        if deferred:
            # Complete through a later step output, so the scheduler sees the
            # failure or invalid blocks first.
            self._defer_kv_connector_outputs(
                [KVConnectorOutput(finished_recving=deferred)]
            )
        finished = set[str]()
        self._count_finished(req_ids - deferred, self._recv_remaining_count, finished)
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
