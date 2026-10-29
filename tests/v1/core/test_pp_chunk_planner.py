# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the equal-time PP prefill step planner (CPU only)."""

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
# Every chunk also pays for its prefix, as MLA does to up-project cached KV.
MLA_COST = ChunkCostModel(o=2.0, a=0.03, b=1.3e-6, e=1.5e-4)
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
        pp_chunk_work_ms=None,
        pp_chunk_overhead_ms=None,
        pp_chunk_last_step_ms=None,
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
    # A budget of exactly an aligned chunk's work yields that chunk.
    for L in range(0, 131072, 977):
        assert planner.max_chunk(COST.work(2048, L), L, 1 << 20) == 2048


def test_target_time_starts_from_prefix():
    """Planning starts from the computed prefix (e.g. a prefix-cache hit)."""
    planner = make_planner(4)
    N = 128 * 1024
    assert planner.target_time(N, 65536) < planner.target_time(N, 0)
    assert planner.target_time(N, N) == COST.o


@pytest.mark.parametrize("cost", [COST, MLA_COST])
def test_fit_recovers_coefficients(cost: ChunkCostModel):
    rng = np.random.default_rng(0)
    samples = []
    for L in (0, 16384, 65536):
        for c in (512, 1024, 2048, 4096, 8192, 16384):
            t = cost.overhead(L) + cost.work(c, L)
            samples.append((c, L, t * (1 + 0.03 * rng.standard_normal())))
    fitted, fit_error = fit_chunk_cost(samples)
    assert fitted.a == pytest.approx(cost.a, rel=0.05)
    assert fitted.b == pytest.approx(cost.b, rel=0.05)
    assert abs(fitted.o - cost.o) <= 0.5
    # The 3% noise is ~2 ms on the shortest chunk at the longest prefix, where
    # MLA_COST's prefix cost is ~10 ms.
    assert abs(fitted.e - cost.e) * 65536 <= 3.0
    assert fit_error <= 0.05
    if cost is MLA_COST:
        return

    # A negative intercept would push T* to zero; o is floored instead.
    shifted = [(c, L, t - COST.o - 1.0) for c, L, t in samples]
    cost, _ = fit_chunk_cost(shifted)
    assert cost.o == pytest.approx(0.01 * np.median([t for *_, t in shifted]))
    assert cost.a >= 0 and cost.b >= 0

    with pytest.raises(ValueError, match="distinct"):
        fit_chunk_cost([(c, 0, t) for c, _, t in samples])


def test_step_target_treats_queue_as_batch():
    """T is taken for the queued prefills served as one FCFS batch: each
    one's TTFT is its work plus the work ahead of it."""
    planner = make_planner(4)
    N = 64 * 1024
    queue = [_req(N), _req(N, 1024), _req(N)]
    cum = np.cumsum([COST.total_work(N, r.num_computed_tokens) for r in queue])
    expected = COST.o + math.sqrt(cum.mean() * COST.o / 3)
    assert planner.step_target_time(queue, _computed) == pytest.approx(expected)
    # A lone prefill gets its own optimum; one token left is a decode.
    lone = planner.target_time(N, 0)
    assert planner.step_target_time([_req(N)], _computed) == pytest.approx(lone)
    assert planner.step_target_time(
        [_req(N), _req(N, N - 1)], _computed
    ) == pytest.approx(lone)

    def endless(counter):
        while True:
            counter.append(None)
            yield _req(N)

    consumed: list[None] = []
    planner.step_target_time(endless(consumed), _computed)
    assert len(consumed) == 64


def test_step_target_scaled_by_measured_queueing():
    """Finished prefills that queued scale the queue's T up by the ratio of
    their time until the last chunk left the first stage to their compute."""
    planner = make_planner(4)
    o, S, N = COST.o, 4, 64 * 1024
    queue = [_req(N)]
    W = COST.total_work(N, 0)
    # TTFT 400 ms, of which 3 * 20 ms crossing the later stages: 340 ms until
    # the last chunk left the first stage, for 100 ms of own compute.
    for _ in range(40):
        planner.record_first_token(400.0, 20.0, 100.0)
    expected = o + math.sqrt(o * W * 3.4 / (S - 1))
    assert planner.step_target_time(queue, _computed) == pytest.approx(expected)
    # Only the last 32 count, and the ratio never drops below 1.
    for _ in range(32):
        planner.record_first_token(100.0, 20.0, 100.0)
    lone = planner.target_time(N, 0)
    assert planner.step_target_time(queue, _computed) == pytest.approx(lone)


def test_step_charges_each_prefill_its_prefix_cost():
    """With a per-chunk prefix cost, a tail behind a long prefix gets fewer
    chunks, and each prefill packed into a step pays its own prefix cost, so
    the step stays within its target."""
    mla_planner = PPChunkPlanner(MLA_COST, 4, GRAN, C_MIN, C_MAX)
    L = 65536
    N = L + 16384
    assert len(plan(mla_planner, N, L)) < len(plan(make_planner(4), N, L))

    T = 300.0
    step = PPChunkStep(mla_planner, T)
    packed = []
    for req in (_req(L + 1024, L), _req(L + 4096, L)):
        c = step.cap(req, L, req.num_tokens - L)
        step.commit(req, L, c)
        packed.append(c)
    assert packed[0] == 1024 and packed[1] > 0
    step_ms = MLA_COST.o + sum(MLA_COST.e * L + MLA_COST.work(c, L) for c in packed)
    assert step_ms <= T


def test_step_records_work_and_last_step():
    """The first chunk records the request's work from where it started, and
    the chunk that completes it records the step time. A preempted request
    recomputes its outputs as prefill too."""
    planner = make_planner(4)
    N, prompt = 100_000, 60_000
    req = _req(N, prompt, num_prompt_tokens=prompt)
    T = planner.target_time(N, prompt)
    step = PPChunkStep(planner, T)
    cap = step.cap(req, prompt, C_MAX)
    assert cap == plan(planner, N, prompt)[0] < C_MAX
    step.commit(req, prompt, cap)
    assert req.pp_chunk_work_ms == pytest.approx(COST.total_work(N, prompt))
    assert req.pp_chunk_last_step_ms is None

    L = N - 512
    step = PPChunkStep(planner, T)
    step.commit(req, L, step.cap(req, L, 512))
    assert req.pp_chunk_work_ms == pytest.approx(COST.total_work(N, prompt))
    assert req.pp_chunk_last_step_ms == T
