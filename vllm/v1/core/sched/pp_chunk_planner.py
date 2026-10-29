# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Equal-time prefill steps for pipeline parallelism.

Every step of prefill chunks is filled up to the same time ``T``, whichever
requests the chunks come from, so no step holds up the pipeline. This module
picks ``T`` from the prefills queued now and how long recent ones queued, and
converts it into chunk sizes using the ``pp_dynamic_chunk_cost`` model:

    d(c, L) = o + e * L + a * c + b * c * (L + c / 2)

where ``c`` is the chunk size, ``L`` the prefix already computed, ``o`` the
per-step fixed overhead, ``e`` the per-chunk cost of each prefix token (e.g.
MLA up-projecting the cached latent KV), ``a`` the per-token cost and ``b``
the cost per (query, key) pair of causal attention.
"""

import json
import math
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.request import Request

# Chunk sizes are rounded down to a multiple of this. Prefill kernels (e.g.
# TRTLLM attention on B300) run 10-30% slower on chunks that are not a
# multiple of 256, so finer alignment loses more than it gains.
PP_CHUNK_ALIGNMENT = 512
# Recently finished prefills whose queueing scales the step time.
_QUEUEING_WINDOW = 32
# Bounds the per-step cost of scanning the queue.
_MAX_QUEUE_SCAN = 64


@dataclass(frozen=True)
class ChunkCostModel:
    """Bottleneck-stage forward time of one prefill chunk, in milliseconds."""

    o: float
    """Fixed overhead per chunk (ms)."""
    a: float
    """Cost per token (ms/token)."""
    b: float
    """Cost per (query, key) pair of causal attention (ms)."""
    e: float = 0.0
    """Cost per prefix token, paid again by every chunk (ms/token)."""

    def overhead(self, L: int) -> float:
        """Part of ``d(c, L)`` that does not depend on ``c``."""
        return self.o + self.e * L

    def work(self, c: int, L: int) -> float:
        """Part of ``d(c, L)`` that grows with ``c``."""
        return self.a * c + self.b * c * (L + c / 2)

    def total_work(self, N: int, L0: int) -> float:
        """Compute needed to prefill positions ``[L0, N)``, however chunked."""
        return self.a * (N - L0) + self.b * (N * N - L0 * L0) / 2

    @classmethod
    def from_json(cls, text: str) -> "ChunkCostModel":
        try:
            data = json.loads(text)
            model = cls(
                o=float(data["o"]),
                a=float(data["a"]),
                b=float(data["b"]),
                e=float(data.get("e", 0.0)),
            )
        except (ValueError, TypeError, KeyError, AttributeError) as e:
            raise ValueError(
                "pp_dynamic_chunk_cost must be a JSON object with numeric "
                f'"o", "a", "b" and optionally "e" keys, got {text!r}'
            ) from e
        if (
            model.o <= 0
            or model.a < 0
            or model.b < 0
            or model.e < 0
            or model.a + model.b <= 0
        ):
            raise ValueError(
                "pp_dynamic_chunk_cost needs o > 0, a >= 0, b >= 0, e >= 0 "
                f"and a + b > 0, got {text!r}"
            )
        return model


class PPChunkPlanner:
    """Picks the time of every prefill step and sizes chunks to fit it.

    Args:
        cost: fitted chunk cost model.
        pp_size: number of pipeline stages ``S``.
        gran: chunk sizes are rounded down to a multiple of this.
        c_min: smallest chunk worth scheduling; keeps GEMMs out of the
            bandwidth-bound regime.
        c_max: largest chunk (the scheduler's token budget).

    """

    def __init__(
        self, cost: ChunkCostModel, pp_size: int, gran: int, c_min: int, c_max: int
    ) -> None:
        assert gran >= 1 and c_min >= 1 and c_max >= 1
        self.cost = cost
        self.pp_size = pp_size
        self.gran = gran
        self.c_min = c_min
        self.c_max = c_max
        # (time until the last chunk left the first stage, own compute), in
        # ms, of recently finished prefills.
        self._finished: deque[tuple[float, float]] = deque(maxlen=_QUEUEING_WINDOW)

    def record_first_token(
        self, ttft_ms: float, last_step_ms: float, work_ms: float
    ) -> None:
        """Record a prefill that produced its first token.

        Args:
            ttft_ms: time since the scheduler received the request.
            last_step_ms: step time when its last chunk was scheduled; that
                chunk spent ``(S - 1) * last_step_ms`` crossing the later
                stages.
            work_ms: its compute from where it started, excluding overheads.

        """
        queued_ms = ttft_ms - (self.pp_size - 1) * last_step_ms
        self._finished.append((queued_ms, work_ms))

    def target_time(self, N: int, L0: int) -> float:
        """Step time (ms) for a lone ``N``-token prompt starting at prefix ``L0``.

        Every step spends ``o`` not computing, and the last chunk takes
        ``(S - 1) * T`` to cross the later stages, so the TTFT
        ``W + W * o / (T - o) + (S - 1) * T`` of ``W`` ms of compute is
        minimized by ``T = o + sqrt(W * o / (S - 1))``, with ``o`` the fixed
        cost of a chunk at the middle of the prompt.
        """
        if self.pp_size <= 1:
            return math.inf
        if N <= L0:
            return self.cost.overhead(L0)
        W = self.cost.total_work(N, L0)
        o = self.cost.overhead((L0 + N) // 2)
        return o + math.sqrt(W * o / (self.pp_size - 1))

    def step_target_time(
        self,
        queue: Iterable["Request"],
        computed_tokens: Callable[["Request"], int],
    ) -> float:
        """Time (ms) every step of prefill chunks should take.

        A step's fixed cost ``o`` delays every request in the system, and
        ``(S - 1) * T`` is added to each TTFT for the last chunk to cross the
        later stages. Balancing the two gives ``T = o + sqrt(o * R / (S - 1))``
        for a mean TTFT ``R``, taken from the prefills queued now served as
        one FCFS batch: each one's TTFT is its own compute plus that of the
        prefills ahead of it, as in ``target_time``. Being the current queue,
        it follows load changes at the next step.

        When arrivals do not wait on service, faster steps also shorten the
        queue, which raises the sensitivity of ``R`` to the step overhead by
        the ratio of a request's time until its last chunk leaves the first
        stage to its own compute. That ratio (at least 1) can only be
        measured, so it is averaged over recently finished prefills and
        scales the square root.

        ``computed_tokens`` gives the prefix a request starts from, so that
        waiting requests can be credited with their expected prefix-cache hits.
        """
        if self.pp_size <= 1:
            return math.inf
        R, o = self._queue_as_batch(queue, computed_tokens)
        ratio = 1.0
        if self._finished:
            queued = sum(q for q, _ in self._finished)
            work = sum(w for _, w in self._finished)
            if work > 0:
                ratio = max(1.0, queued / work)
        return o + math.sqrt(o * R * ratio / (self.pp_size - 1))

    def _queue_as_batch(
        self,
        queue: Iterable["Request"],
        computed_tokens: Callable[["Request"], int],
    ) -> tuple[float, float]:
        """Mean TTFT compute and work-weighted fixed cost per chunk of the
        queued prefills, served first-come first-served."""
        cost = self.cost
        num_reqs = 0
        cum_work = 0.0
        sum_cum_work = 0.0
        sum_overhead_work = 0.0
        for request in queue:
            L = computed_tokens(request)
            if not _is_prefill(request, L):
                continue
            work, overhead = request.pp_chunk_work_ms, request.pp_chunk_overhead_ms
            if work is None or overhead is None:
                work = cost.total_work(request.num_tokens, L)
                overhead = cost.overhead((L + request.num_tokens) // 2)
            cum_work += work
            sum_cum_work += cum_work
            sum_overhead_work += work * overhead
            num_reqs += 1
            if num_reqs == _MAX_QUEUE_SCAN:
                break
        if cum_work <= 0:
            return 0.0, cost.o
        return sum_cum_work / num_reqs, sum_overhead_work / cum_work

    def max_chunk(self, work_budget: float, L: int, remaining: int) -> int:
        """Largest chunk at prefix ``L`` whose compute fits ``work_budget`` ms.

        Rounded down to ``gran`` and capped at ``min(c_max, remaining)``. The
        ``c_min`` floor is applied by the caller. A tail shorter than ``gran``
        is returned whole rather than rounded to zero.
        """
        if work_budget <= 0 or remaining <= 0:
            return 0
        limit = min(self.c_max, remaining)
        a, b = self.cost.a, self.cost.b
        lin = a + b * L
        if b > 0:
            # Equivalent to (-lin + sqrt(lin^2 + 2 b B)) / b without
            # cancellation when b is tiny.
            c = 2 * work_budget / (lin + math.sqrt(lin * lin + 2 * b * work_budget))
        elif a > 0:
            c = work_budget / a
        else:
            return limit
        if c >= limit:
            return limit
        # The slack keeps a budget of exactly an aligned chunk's work from
        # rounding down a whole granule on float error.
        aligned = int(c + 1e-6) // self.gran * self.gran
        if aligned == 0 and remaining <= self.gran:
            return remaining
        return aligned


class PPChunkStep:
    """Caps the prefill chunks of one scheduling step to ``target_ms``.

    Chunks are taken in scheduling order until the step's predicted time
    reaches the target. The step pays ``o`` once and every prefill chunk its
    own ``e * L``. The first prefill chunk always gets at least ``c_min``
    tokens.
    """

    def __init__(self, planner: PPChunkPlanner, target_ms: float) -> None:
        self.planner = planner
        self.target_ms = target_ms
        self.work_used = 0.0
        self.has_prefill = False

    def cap(
        self, request: "Request", num_computed_tokens: int, num_new_tokens: int
    ) -> int:
        """Return the capped chunk size, 0 to leave the request for a later step.

        Only prefills are capped; decodes, including spec-decode padded ones,
        pass through unchanged.
        """
        if not _is_prefill(request, num_computed_tokens):
            return num_new_tokens
        planner = self.planner
        budget = (
            self.target_ms - planner.cost.overhead(num_computed_tokens) - self.work_used
        )
        remaining = request.num_tokens - num_computed_tokens
        cap = planner.max_chunk(budget, num_computed_tokens, remaining)
        floor = min(planner.c_min, remaining)
        if not self.has_prefill:
            cap = max(cap, floor)
        elif cap < floor:
            return 0
        return min(num_new_tokens, cap)

    def commit(
        self, request: "Request", num_computed_tokens: int, num_new_tokens: int
    ) -> None:
        """Charge a scheduled chunk against the step's budget."""
        cost = self.planner.cost
        self.work_used += cost.work(num_new_tokens, num_computed_tokens)
        if not _is_prefill(request, num_computed_tokens):
            return
        self.work_used += cost.e * num_computed_tokens
        self.has_prefill = True
        N = request.num_tokens
        if request.pp_chunk_work_ms is None:
            request.pp_chunk_work_ms = cost.total_work(N, num_computed_tokens)
            request.pp_chunk_overhead_ms = cost.overhead((num_computed_tokens + N) // 2)
        if num_computed_tokens + num_new_tokens >= N:
            request.pp_chunk_last_step_ms = self.target_ms


def _is_prefill(request: "Request", num_computed_tokens: int) -> bool:
    # Same predicate as the scheduler's prefill cadence: a request with one
    # token left (a full prefix cache hit, a resumed decode) is a decode.
    return num_computed_tokens < request.num_tokens - 1
