# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.early_recv import (
    EarlyRecvReporter,
    EarlyRecvTracker,
)
from vllm.v1.outputs import KVConnectorOutput

from .utils import take_early_reports

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def tracker():
    tracker = EarlyRecvTracker(num_workers=2)
    yield tracker
    tracker.close()


@pytest.fixture
def reporters(tracker):
    reporters = [EarlyRecvReporter(rank) for rank in range(2)]
    for reporter in reporters:
        reporter.connect(tracker.path)
    yield reporters
    for reporter in reporters:
        reporter.close()


def _step_output(tracker: EarlyRecvTracker, *req_ids: str) -> set[str]:
    """The finished loads that a step output reporting `req_ids` passes on."""
    output = KVConnectorOutput(finished_recving=set(req_ids))
    tracker.update_connector_output(output)
    return output.finished_recving


def test_load_finishes_once_every_worker_reports(tracker, reporters):
    tracker.add_load("a")
    reporters[0].report(["a"])
    reporters[0].report(["a"])
    assert take_early_reports(tracker, reporters) == set()
    reporters[1].report(["a"])
    assert take_early_reports(tracker, reporters) == {"a"}
    assert tracker.awaits_step_output("a")

    assert _step_output(tracker, "a", "b") == {"b"}
    assert not tracker.awaits_step_output("a")
    for reporter in reporters:
        reporter.report(["a"])
    assert take_early_reports(tracker, reporters) == set()


def test_step_output_first_ignores_later_reports(tracker, reporters):
    tracker.add_load("a")
    reporters[0].report(["a"])
    assert _step_output(tracker, "a") == {"a"}
    reporters[1].report(["a"])
    assert take_early_reports(tracker, reporters) == set()


def test_finished_request_load_goes_through_step_output(tracker, reporters):
    """A request that finished while loading frees its blocks when its step
    output arrives, so the load is not reported early."""
    tracker.add_load("a")
    reporters[0].report(["a"])
    tracker.request_finished("a")
    reporters[1].report(["a"])
    assert take_early_reports(tracker, reporters) == set()
    assert _step_output(tracker, "a") == {"a"}


def test_request_finished_after_early_report_keeps_dropping_step_output(
    tracker, reporters
):
    """The scheduler freed a resumed request when it finished, so its later
    step output must not report the load again."""
    tracker.add_load("a")
    for reporter in reporters:
        reporter.report(["a"])
    assert take_early_reports(tracker, reporters) == {"a"}
    tracker.request_finished("a")
    assert _step_output(tracker, "a") == set()


def test_reports_to_an_absent_scheduler_do_not_block(tmp_path):
    reporter = EarlyRecvReporter(rank=0)
    reporter.connect(f"ipc://{tmp_path}/absent")
    for i in range(4096):
        reporter.report([str(i)])
    reporter.close()
