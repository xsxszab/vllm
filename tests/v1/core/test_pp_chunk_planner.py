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
    fit_chunk_cost,
)

# Placeholder coefficients from the design doc: o=2 ms, a=0.03 ms/token,
# b=1.3e-6 ms per (q, k) pair.
COST = ChunkCostModel(o=2.0, a=0.03, b=1.3e-6)
GRAN = 64
C_MIN = 512
C_MAX = 32768


def make_planner(pp_size: int, cost: ChunkCostModel = COST) -> PPChunkPlanner:
    return PPChunkPlanner(cost, pp_size, GRAN, C_MIN, C_MAX)


def simulate(chunks: list[int], cost: ChunkCostModel, pp_size: int, L0: int = 0):
    """Max-plus flow-shop recursion: TTFT of `chunks` through `pp_size` equal
    stages with no communication cost."""
    durations = []
    L = L0
    for c in chunks:
        durations.append(cost.predict(c, L))
        L += c
    done = [0.0] * (pp_size + 1)  # done[s]: finish time of last chunk on stage s
    for d in durations:
        for s in range(1, pp_size + 1):
            done[s] = max(done[s], done[s - 1]) + d
    return done[pp_size]


@pytest.mark.parametrize("N", [32 * 1024, 128 * 1024, 512 * 1024])
@pytest.mark.parametrize("pp_size", [2, 4, 8])
def test_closed_form_target_is_near_optimal(N: int, pp_size: int):
    """The closed-form T* must land within 2% of the best T found by brute
    force over a grid, under the max-plus recursion."""
    planner = make_planner(pp_size)
    ttft = simulate(planner.plan(N, 0), COST, pp_size)

    t_lo = COST.predict(C_MIN, 0)
    t_hi = COST.predict(C_MAX, 0)
    best = min(
        simulate(planner.plan(N, 0, T), COST, pp_size)
        for T in np.geomspace(t_lo, t_hi, 200)
    )
    assert ttft <= best * 1.02
    # And it must beat a fixed 8K chunking for long prompts.
    if N >= 128 * 1024:
        fixed = simulate([8192] * (N // 8192), COST, pp_size)
        assert ttft < fixed


def test_chunks_take_equal_time():
    """Every chunk except those pinned by c_min/c_max or the tail lands within
    one alignment granule below the target."""
    planner = make_planner(4)
    N = 128 * 1024
    T = planner.target_time(N, 0)
    chunks = planner.plan(N, 0, T)
    assert sum(chunks) == N
    L = 0
    for c in chunks[:-1]:
        d = COST.predict(c, L)
        if c == C_MAX:
            assert d <= T + 1e-9
        elif c == C_MIN:
            # c_min floor: may exceed the target but no smaller chunk exists.
            assert d >= T - COST.work(GRAN, L + c) - 1e-9
        else:
            assert d <= T + 1e-9
            assert T - d <= COST.work(GRAN, L + c) + 1e-9
        L += c
    # Prefix grows, so chunks shrink.
    assert all(a >= b for a, b in zip(chunks, chunks[1:-1]))


def test_predict_ttft_matches_simulation():
    planner = make_planner(4)
    N = 128 * 1024
    T = planner.target_time(N, 0)
    assert simulate(planner.plan(N, 0, T), COST, 4) == pytest.approx(
        planner.predict_ttft(N, 0, T), rel=0.05
    )


def test_max_chunk_monotonic():
    planner = make_planner(4)
    budgets = [10.0, 50.0, 100.0, 400.0]
    prefixes = [0, 4096, 65536, 500_000]
    for B in budgets:
        caps = [planner.max_chunk(B, L, 1 << 20) for L in prefixes]
        assert caps == sorted(caps, reverse=True)
    for L in prefixes:
        caps = [planner.max_chunk(B, L, 1 << 20) for B in budgets]
        assert caps == sorted(caps)


def test_max_chunk_boundaries():
    planner = make_planner(4)
    assert planner.max_chunk(0.0, 0, 1000) == 0
    assert planner.max_chunk(-1.0, 0, 1000) == 0
    assert planner.max_chunk(100.0, 0, 0) == 0
    # Aligned down to gran.
    assert planner.max_chunk(100.0, 0, 1 << 20) % GRAN == 0
    # A tail shorter than gran is returned whole.
    assert planner.max_chunk(0.5, 0, 10) == 10
    # Capped at c_max and at the remaining prompt.
    assert planner.max_chunk(1e9, 0, 1 << 20) == C_MAX
    assert planner.max_chunk(1e9, 0, 777) == 777
    # b == 0: linear cost.
    linear = PPChunkPlanner(ChunkCostModel(2.0, 0.1, 0.0), 4, GRAN, C_MIN, C_MAX)
    assert linear.max_chunk(100.0, 1 << 20, 1 << 20) == 1000 // GRAN * GRAN
    # Chunk fits exactly its budget.
    c = planner.max_chunk(100.0, 65536, 1 << 20)
    assert COST.work(c, 65536) <= 100.0 < COST.work(c + GRAN, 65536)


def test_target_time_bounds():
    planner = make_planner(4)
    # The first chunk is at least c_min, so T is at least that chunk's time.
    N = 2048
    assert planner.target_time(N, 0) >= COST.predict(C_MIN, 0)
    # ... and at most c_max.
    assert planner.target_time(1 << 20, 0) <= COST.predict(C_MAX, 0)
    # A remainder shorter than c_min bounds T by the remainder.
    assert planner.target_time(N, N - 100) == pytest.approx(COST.predict(100, N - 100))
    # For a long prompt an existing prefix lowers both W and T.
    N = 128 * 1024
    assert planner.target_time(N, 65536) < planner.target_time(N, 0)
    # Nothing left: only the overhead.
    assert planner.target_time(N, N) == COST.o


def _grid_samples(cost: ChunkCostModel, noise: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    samples = []
    for L in (0, 16384, 65536):
        for c in (512, 1024, 2048, 4096, 8192, 16384):
            t = cost.predict(c, L) * (1 + noise * rng.standard_normal())
            samples.append((c, L, t))
    return samples


def test_fit_recovers_coefficients():
    cost, report = fit_chunk_cost(_grid_samples(COST, noise=0.03))
    assert cost.a == pytest.approx(COST.a, rel=0.05)
    assert cost.b == pytest.approx(COST.b, rel=0.05)
    assert abs(cost.o - COST.o) <= 0.5
    assert report.fit_error <= 0.05
    assert report.num_samples == 18
    assert not report.o_floored


def test_fit_floors_negative_overhead():
    """A negative intercept would push T* to zero; it is clamped to a fraction
    of the median sample instead."""
    samples = [(c, L, COST.work(c, L) - 1.0) for c, L, _ in _grid_samples(COST, 0.0)]
    cost, report = fit_chunk_cost(samples, o_floor_frac=0.01)
    assert report.o_floored
    median = float(np.median([t for _, _, t in samples]))
    assert cost.o == pytest.approx(0.01 * median)
    assert cost.a >= 0 and cost.b >= 0


def test_fit_rejects_insufficient_grid():
    one_chunk_size = [(1024, L, COST.predict(1024, L)) for L in (0, 4096, 8192)]
    with pytest.raises(ValueError, match="distinct"):
        fit_chunk_cost(one_chunk_size)
    one_prefix = [(c, 0, COST.predict(c, 0)) for c in (512, 1024, 2048)]
    with pytest.raises(ValueError, match="distinct"):
        fit_chunk_cost(one_prefix)


def test_cost_model_from_json():
    cost = ChunkCostModel.from_json('{"o": 2, "a": 0.03, "b": 1.3e-6}')
    assert cost == COST
    for bad in (
        "{}",
        "[1,2,3]",
        '{"o": 0, "a": 1, "b": 1}',
        '{"o": 1, "a": -1, "b": 1}',
    ):
        with pytest.raises(ValueError):
            ChunkCostModel.from_json(bad)


def test_total_work_is_chunking_independent():
    N, L0 = 100_000, 1234
    L = L0
    total = 0.0
    for c in make_planner(4).plan(N, L0):
        total += COST.work(c, L)
        L += c
    assert math.isclose(total, COST.total_work(N, L0), rel_tol=1e-9)


def _req(num_tokens: int, num_computed_tokens: int = 0, **kwargs) -> SimpleNamespace:
    return SimpleNamespace(
        request_id="r",
        num_tokens=num_tokens,
        num_computed_tokens=num_computed_tokens,
        pp_chunk_target_ms=None,
        **kwargs,
    )


def test_queue_target_uses_mean_cumulative_work():
    """FCFS backlog: T* is taken for the mean of each request's work plus the
    work queued ahead of it."""
    planner = make_planner(4)
    N = 64 * 1024
    W = COST.total_work(N, 0)
    queue = [_req(N), _req(N, 1024), _req(N)]
    cum = [COST.total_work(N, 0)]
    cum.append(cum[-1] + COST.total_work(N, 1024))
    cum.append(cum[-1] + W)
    expected = COST.o + math.sqrt(sum(cum) / 3 * COST.o / 3)
    assert planner.queue_target_time(queue) == pytest.approx(expected)


def test_queue_target_needs_two_prefills():
    planner = make_planner(4)
    N = 64 * 1024
    assert planner.queue_target_time([]) == 0
    assert planner.queue_target_time([_req(N)]) == 0
    # One token left (full prefix hit, decode) is not queued prefill work.
    assert planner.queue_target_time([_req(N), _req(N, N - 1)]) == 0
    assert make_planner(1).queue_target_time([_req(N), _req(N)]) == 0


def test_queue_target_grows_with_backlog_and_bounds_scan():
    N = 32 * 1024
    planner = PPChunkPlanner(COST, 4, GRAN, C_MIN, 4096)
    targets = [planner.queue_target_time([_req(N)] * k) for k in range(2, 10)]
    assert targets == sorted(targets)

    def endless(counter):
        while True:
            counter.append(None)
            yield _req(N)

    # Stops once the head's chunk alone fills the token budget.
    consumed: list[None] = []
    T = planner.queue_target_time(endless(consumed))
    assert T - COST.o >= COST.work(4096, 0)
    assert len(consumed) < 64
    # Otherwise the scan is bounded.
    consumed = []
    make_planner(4).queue_target_time(endless(consumed))
    assert len(consumed) == 64


def test_step_passes_decodes_through():
    """A decode, including a spec-decode padded first decode step after a
    full prefix hit, is neither capped nor sets the step target."""
    step = PPChunkStep(make_planner(4))
    req = _req(33, 32)
    assert step.cap(req, 32, 4) == 4
    step.commit(req, 32, 4)
    assert step.target_ms is None
    assert req.pp_chunk_target_ms is None


def test_step_caps_resumed_output_tokens():
    """A preempted request recomputes its outputs as prefill too."""
    planner = make_planner(4)
    N, prompt = 100_000, 60_000
    req = _req(N, prompt, num_prompt_tokens=prompt)
    step = PPChunkStep(planner)
    cap = step.cap(req, prompt, C_MAX)
    assert cap == planner.plan(N, prompt)[0] < C_MAX
    step.commit(req, prompt, cap)
    assert req.pp_chunk_target_ms == pytest.approx(planner.target_time(N, prompt))


@pytest.mark.parametrize("fits", [True, False])
def test_step_packs_next_request_within_target(fits: bool):
    """After A's last chunk, B gets the rest of A's target if that holds at
    least a c_min chunk, else waits for the next step."""
    planner = make_planner(4)
    N_a, tail = 100_000, 600
    L_a = N_a - tail
    spare = COST.work(2 * C_MIN, 0) if fits else COST.work(C_MIN, 0) / 2
    a = _req(N_a, L_a)
    a.pp_chunk_target_ms = COST.o + COST.work(tail, L_a) + spare
    b = _req(16 * 1024)

    step = PPChunkStep(planner)
    assert step.cap(a, L_a, tail) == tail
    step.commit(a, L_a, tail)
    cap_b = step.cap(b, 0, 16 * 1024)
    if fits:
        assert cap_b == planner.max_chunk(spare, 0, 16 * 1024) >= C_MIN
    else:
        assert cap_b == 0


def test_step_first_chunk_uses_queue_target():
    planner = make_planner(4)
    N = 128 * 1024
    T = planner.target_time(N, 0)
    for queue_ms, expected_ms in ((0.0, T), (T / 2, T), (2 * T, 2 * T)):
        step = PPChunkStep(planner, queue_target_ms=queue_ms)
        req = _req(N)
        cap = step.cap(req, 0, C_MAX)
        assert cap == planner.max_chunk(expected_ms - COST.o, 0, N)
        step.commit(req, 0, cap)
        assert step.target_ms == pytest.approx(expected_ms)
        # The request keeps its own single-request target.
        assert req.pp_chunk_target_ms == pytest.approx(T)
