# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import zmq

from vllm.distributed.kv_transfer import kv_transfer_state
from vllm.utils.network_utils import get_open_zmq_ipc_path, make_zmq_socket
from vllm.v1.executor.early_kv_recv import EarlyKVOutputAggregator
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput


def test_poll_failure_exits_worker():
    # Exceptions are suppressed so that only os._exit(1) can produce exit code 1.
    code = """
from contextlib import suppress
from unittest.mock import Mock
with suppress(Exception):
    from vllm.distributed.kv_transfer import (
        init_early_recv_reporting,
        report_early_finished_recving,
    )
    from vllm.utils.network_utils import get_open_zmq_ipc_path
    init_early_recv_reporting(get_open_zmq_ipc_path())
    connector = Mock()
    connector.poll_finished_recving.side_effect = RuntimeError("poll")
    report_early_finished_recving(connector)
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 1
    assert "Early KV receive poll failed" in result.stdout + result.stderr


def test_idle_polling_follows_steps_and_pending_loads(monkeypatch: pytest.MonkeyPatch):
    """A worker waiting for its next step polls only while loads are pending
    and a step ran recently; every report or poll sends what finished."""
    now = [100.0]
    monkeypatch.setattr(
        kv_transfer_state, "time", SimpleNamespace(monotonic=lambda: now[0])
    )
    connector = Mock()
    connector.has_pending_recving.return_value = True
    connector.poll_finished_recving.return_value = {"request"}
    path = get_open_zmq_ipc_path()
    engine_end = make_zmq_socket(zmq.Context.instance(), path, zmq.PULL)
    kv_transfer_state.init_early_recv_reporting(path)

    def received() -> set[str]:
        assert engine_end.poll(timeout=5000)
        return engine_end.recv_pyobj()

    try:
        # No step has run yet.
        assert kv_transfer_state.next_early_recv_poll(connector) is None
        kv_transfer_state.report_early_finished_recving(connector)
        assert received() == {"request"}
        deadline = kv_transfer_state.next_early_recv_poll(connector)
        assert deadline is not None and deadline > now[0]
        kv_transfer_state.poll_early_finished_recving(connector)
        assert received() == {"request"}
        connector.has_pending_recving.return_value = False
        assert kv_transfer_state.next_early_recv_poll(connector) is None
        connector.has_pending_recving.return_value = True
        # No step for longer than the active window: the engine is idle.
        now[0] += 2.0
        assert kv_transfer_state.next_early_recv_poll(connector) is None
    finally:
        kv_transfer_state.ensure_kv_transfer_shutdown()
        engine_end.close(linger=0)


@pytest.mark.parametrize(
    "ordinary_rank,failed,invalid",
    [(None, False, False)]
    + [
        (rank, failed, invalid)
        for rank in (0, 1)
        for failed, invalid in ((False, False), (False, True), (True, False))
    ],
)
def test_early_receives_share_normal_quorum(ordinary_rank, failed, invalid):
    aggregator = EarlyKVOutputAggregator(2)

    def aggregate(kv_output=None):
        output = ModelRunnerOutput([], {}, kv_connector_output=kv_output)
        aggregator.aggregate([output])
        assert output.kv_connector_output is not None
        return output.kv_connector_output

    early = set()
    normal = KVConnectorOutput()
    for rank in range(2):
        if rank == ordinary_rank:
            # A step-output vote: a failure, invalid blocks, or a success that
            # was not reported early.
            normal = aggregate(
                KVConnectorOutput(
                    finished_recving={"request"},
                    failed_recving={"request"} if failed else set(),
                    invalid_block_ids={7} if invalid else set(),
                ),
            )
        else:
            early |= aggregator.add_finished_recving({"request"})
        if rank == 0:
            assert not early and not normal.finished_recving
    # A step without output must keep a deferred vote for the next one.
    assert aggregator.aggregate([None]) is None
    if not normal.finished_recving:
        normal = aggregate()
    # A step output with a vote may carry invalid blocks or a failure that the
    # scheduler has not processed yet, so a receive finishes early only when
    # every worker reported it early.
    assert early == ({"request"} if ordinary_rank is None else set())
    assert (normal.finished_recving or set()) == {"request"} - early
    assert normal.failed_recving == ({"request"} if failed else set())
    assert aggregate().is_empty()
