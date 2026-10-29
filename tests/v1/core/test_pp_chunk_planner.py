# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the equal-time PP prefill chunk planner (CPU only)."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from vllm.v1.core.sched.pp_chunk_planner import (
    ChunkCostModel,
    PPChunkPlanner,
    PPChunkStep,
)
from vllm.v1.core.sched.pp_chunk_profile import fit_chunk_cost

from .utils import pp_chunk_plan as plan

COST = ChunkCostModel(o=2.0, a=0.03, b=1.3e-6)
GRAN = 64
C_MIN = 512
C_MAX = 32768


def make_planner(pp_size: int, c_max: int = C_MAX) -> PPChunkPlanner:
    return PPChunkPlanner(COST, pp_size, GRAN, C_MIN, c_max)


def predict(c: int, L: int) -> float:
    return COST.o + COST.work(c, L)


def simulate(chunks: list[int], pp_size: int) -> float:
    """TTFT of `chunks` through `pp_size` equal stages (max-plus recursion)."""
    done = [0.0] * (pp_size + 1)
    L = 0
    for c in chunks:
        d = predict(c, L)
        L += c
        for s in range(1, pp_size + 1):
            done[s] = max(done[s], done[s - 1]) + d
    return done[pp_size]


def _req(num_tokens: int, num_computed_tokens: int = 0, **kwargs) -> SimpleNamespace:
    return SimpleNamespace(
        num_tokens=num_tokens,
        num_computed_tokens=num_computed_tokens,
        pp_chunk_target_ms=None,
        **kwargs,
    )


def _computed(request) -> int:
    return request.num_computed_tokens


@pytest.mark.parametrize("N", [32 * 1024, 128 * 1024, 512 * 1024])
@pytest.mark.parametrize("pp_size", [2, 4, 8])
def test_closed_form_target_is_near_optimal(N: int, pp_size: int):
    """The closed-form T* lands within 2% of the best T found by brute force,
    and beats a fixed 8K chunking for long prompts."""
    planner = make_planner(pp_size)
    ttft = simulate(plan(planner, N, 0), pp_size)
    best = min(
        simulate(plan(planner, N, 0, T), pp_size)
        for T in np.geomspace(predict(C_MIN, 0), predict(C_MAX, 0), 200)
    )
    assert ttft <= best * 1.02
    if N >= 128 * 1024:
        assert ttft < simulate([8192] * (N // 8192), pp_size)


def test_chunks_take_equal_time():
    """Chunks not pinned by c_min or the tail land within one alignment
    granule below the target, and shrink as the prefix grows."""
    planner = make_planner(4)
    N = 128 * 1024
    T = planner.target_time(N, 0)
    chunks = plan(planner, N, 0, T)
    assert sum(chunks) == N
    L = 0
    for c in chunks[:-1]:
        if c > C_MIN:
            assert 0 <= T - predict(c, L) <= COST.work(GRAN, L + c) + 1e-9
        L += c
    assert all(a >= b for a, b in zip(chunks, chunks[1:-1]))


def test_max_chunk_boundaries():
    planner = make_planner(4)
    assert planner.max_chunk(0.0, 0, 1000) == 0
    assert planner.max_chunk(100.0, 0, 0) == 0
    # The largest aligned chunk that fits the budget.
    c = planner.max_chunk(100.0, 65536, 1 << 20)
    assert c % GRAN == 0
    assert COST.work(c, 65536) <= 100.0 < COST.work(c + GRAN, 65536)
    # A tail shorter than gran is returned whole.
    assert planner.max_chunk(0.5, 0, 10) == 10
    # Capped at c_max and at the remaining prompt.
    assert planner.max_chunk(1e9, 0, 1 << 20) == C_MAX
    assert planner.max_chunk(1e9, 0, 777) == 777


def test_target_time_bounds():
    planner = make_planner(4)
    N = 128 * 1024
    assert planner.target_time(2048, 0) >= predict(C_MIN, 0)
    assert planner.target_time(1 << 20, 0) <= predict(C_MAX, 0)
    # A remainder shorter than c_min bounds T by the remainder.
    assert planner.target_time(2048, 1948) == pytest.approx(predict(100, 1948))
    # Planning starts from the computed prefix (e.g. a prefix-cache hit).
    assert planner.target_time(N, 65536) < planner.target_time(N, 0)
    assert planner.target_time(N, N) == COST.o


def test_fit_recovers_coefficients():
    rng = np.random.default_rng(0)
    samples = [
        (c, L, predict(c, L) * (1 + 0.03 * rng.standard_normal()))
        for L in (0, 16384, 65536)
        for c in (512, 1024, 2048, 4096, 8192, 16384)
    ]
    cost, fit_error = fit_chunk_cost(samples)
    assert cost.a == pytest.approx(COST.a, rel=0.05)
    assert cost.b == pytest.approx(COST.b, rel=0.05)
    assert abs(cost.o - COST.o) <= 0.5
    assert fit_error <= 0.05

    # A negative intercept would push T* to zero; o is floored instead.
    shifted = [(c, L, t - COST.o - 1.0) for c, L, t in samples]
    cost, _ = fit_chunk_cost(shifted)
    assert cost.o == pytest.approx(0.01 * np.median([t for *_, t in shifted]))
    assert cost.a >= 0 and cost.b >= 0

    with pytest.raises(ValueError, match="distinct"):
        fit_chunk_cost([(c, 0, t) for c, _, t in samples])


def test_queue_target_uses_mean_cumulative_work():
    """FCFS backlog: T* is taken for the mean of each request's work plus the
    work queued ahead of it."""
    planner = make_planner(4)
    N = 64 * 1024
    queue = [_req(N), _req(N, 1024), _req(N)]
    cum = np.cumsum([COST.total_work(N, r.num_computed_tokens) for r in queue])
    expected = COST.o + math.sqrt(cum.mean() * COST.o / 3)
    assert planner.queue_target_time(queue, _computed) == pytest.approx(expected)
    # A lone prefill keeps its own target; one token left is a decode.
    assert planner.queue_target_time([_req(N)], _computed) == 0
    assert planner.queue_target_time([_req(N), _req(N, N - 1)], _computed) == 0


def test_queue_target_grows_with_backlog_and_bounds_scan():
    N = 32 * 1024
    planner = make_planner(4, c_max=4096)
    targets = [planner.queue_target_time([_req(N)] * k, _computed) for k in (2, 4, 8)]
    assert targets == sorted(targets)

    def endless(counter):
        while True:
            counter.append(None)
            yield _req(N)

    # Stops once the head's chunk alone fills the token budget.
    consumed: list[None] = []
    T = planner.queue_target_time(endless(consumed), _computed)
    assert T - COST.o >= COST.work(4096, 0)
    assert len(consumed) < 64
    # Otherwise the scan is bounded.
    consumed = []
    make_planner(4).queue_target_time(endless(consumed), _computed)
    assert len(consumed) == 64


def test_step_caps_resumed_output_tokens():
    """A preempted request recomputes its outputs as prefill too."""
    planner = make_planner(4)
    N, prompt = 100_000, 60_000
    req = _req(N, prompt, num_prompt_tokens=prompt)
    step = PPChunkStep(planner)
    cap = step.cap(req, prompt, C_MAX)
    assert cap == plan(planner, N, prompt)[0] < C_MAX
    step.commit(req, prompt, cap)
    assert req.pp_chunk_target_ms == pytest.approx(planner.target_time(N, prompt))
