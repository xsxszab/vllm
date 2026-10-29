# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Equal-time prefill chunking for pipeline parallelism.

A prompt split into chunks of durations ``d_1..d_K`` flows through ``S``
pipeline stages as a proportionate flow shop, so its TTFT is
``sum(d_k) + (S - 1) * max(d_k)``. The scheduler therefore sizes every chunk
to the same target duration ``T``, and this module picks ``T`` (raised under
a backlog of prefills, for throughput) and converts it into token counts
using the ``pp_dynamic_chunk_cost`` model:

    d(c, L) = o + e * L + a * c + b * c * (L + c / 2)

where ``c`` is the chunk size, ``L`` the prefix already computed, ``o`` the
per-chunk fixed overhead, ``e`` the per-chunk cost of each prefix token (e.g.
MLA up-projecting the cached latent KV), ``a`` the per-token cost and ``b``
the cost per (query, key) pair of causal attention.
"""

import json
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.request import Request

# Chunk sizes are rounded down to a multiple of this. Prefill kernels (e.g.
# TRTLLM attention on B300) run 10-30% slower on chunks that are not a
# multiple of 256, so finer alignment loses more than it gains.
PP_CHUNK_ALIGNMENT = 512
# Bounds the per-step cost of PPChunkPlanner.queue_target_time.
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
    """Turns the cost model into a per-request target time and chunk sizes.

    Args:
        cost: fitted chunk cost model.
        pp_size: number of pipeline stages ``S``.
        gran: chunk sizes are rounded down to a multiple of this.
        c_min: smallest chunk worth scheduling; keeps GEMMs out of the
            bandwidth-bound regime.
        c_max: largest chunk (the scheduler's token budget).
        backlog_tokens: with a backlog of prefills, steps take at least as
            long as a chunk of this many tokens.

    """

    def __init__(
        self,
        cost: ChunkCostModel,
        pp_size: int,
        gran: int,
        c_min: int,
        c_max: int,
        backlog_tokens: int,
    ) -> None:
        assert gran >= 1 and c_min >= 1 and c_max >= 1 and backlog_tokens >= 1
        self.cost = cost
        self.pp_size = pp_size
        self.gran = gran
        self.c_min = c_min
        self.c_max = c_max
        self.backlog_tokens = backlog_tokens

    def target_time(self, N: int, L0: int) -> float:
        """Target duration ``T`` (ms) of every chunk of an ``N``-token prompt.

        ``T* = o + sqrt(W * o / (S - 1))`` minimizes
        ``W + W * o / (T - o) + (S - 1) * T``, with ``o`` the per-chunk
        overhead at the middle of the prompt; it is then clamped so the
        first chunk is at least ``c_min`` and at most ``c_max`` tokens.
        """
        cost = self.cost
        remaining = max(N - L0, 0)
        if remaining == 0:
            return cost.overhead(L0)
        if self.pp_size > 1:
            W = cost.total_work(N, L0)
            o = cost.overhead((L0 + N) // 2)
            T = o + math.sqrt(W * o / (self.pp_size - 1))
        else:
            T = math.inf
        first = cost.overhead(L0)
        T = max(T, first + cost.work(min(self.c_min, remaining), L0))
        T = min(T, first + cost.work(min(self.c_max, remaining), L0))
        return T

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

    def queue_target_time(
        self,
        queue: Iterable["Request"],
        computed_tokens: Callable[["Request"], int],
    ) -> tuple[float, int]:
        """Step target (ms) minimizing the mean TTFT of the queued prefills.

        Under FCFS with equal-time steps, the ``j``-th queued prefill gets its
        first token after ``W_j + W_j * o / (T - o) + (S - 1) * T``, where
        ``W_j`` is its remaining work plus that of every prefill ahead of it.
        The mean TTFT is therefore minimized by ``target_time``'s formula with
        ``W`` replaced by the mean of ``W_j`` and ``o`` by the work-weighted
        mean overhead, which grows with the backlog and so trades the pipeline
        fill of the head request for throughput.

        ``computed_tokens`` gives the prefix each request starts from, so that
        waiting requests can be credited with their expected prefix-cache
        hits. The scan stops once the head request's chunk alone would fill
        the token budget.

        Returns:
            The target, 0 with fewer than two queued prefills (leaving a lone
            request at its own target), and the number of prefills scanned.

        """
        if self.pp_size <= 1:
            return 0.0, 0
        cost = self.cost
        num_reqs = 0
        cum_work = 0.0
        sum_cum_work = 0.0
        sum_overhead_work = 0.0
        full_step_work = math.inf
        for request in queue:
            L = computed_tokens(request)
            if not _is_prefill(request, L):
                continue
            N = request.num_tokens
            if num_reqs == 0:
                full_step_work = cost.work(self.c_max, L)
            W = cost.total_work(N, L)
            cum_work += W
            sum_cum_work += cum_work
            sum_overhead_work += W * cost.overhead((L + N) // 2)
            num_reqs += 1
            o = sum_overhead_work / cum_work
            if (
                num_reqs == _MAX_QUEUE_SCAN
                or sum_cum_work / num_reqs * o / (self.pp_size - 1) >= full_step_work**2
            ):
                break
        if num_reqs < 2:
            return 0.0, num_reqs
        mean_work = sum_cum_work / num_reqs
        return o + math.sqrt(mean_work * o / (self.pp_size - 1)), num_reqs

    def backlog_target_time(self, num_prefills: int, L: int) -> float:
        """Lower bound (ms) on the step target with ``num_prefills`` queued.

        The queue target assumes no further arrivals; under sustained load,
        the throughput lost to small chunks delays every later request too,
        so each step is at least as long as a ``backlog_tokens`` chunk at the
        prefix ``L`` of the scheduled request. With up to ``pp_size``
        prefills, that chunk's work is costed at prefix 0, so that short tails
        after long cached prefixes still get split across the stages; with
        more, distinct requests keep every stage busy.
        """
        if num_prefills < 2:
            return 0.0
        c = min(self.backlog_tokens, self.c_max)
        cost = self.cost
        return cost.overhead(L) + cost.work(c, L if num_prefills > self.pp_size else 0)


class PPChunkStep:
    """Caps the prefill chunks of one scheduling step to a shared target.

    The first prefill chunk scheduled in a step sets the step's time target:
    the largest of its request's own ``T``, ``queue_target_ms`` and the
    planner's backlog target for ``num_queued_prefills``.
    Later requests only get what is left of it, so a chunk that finishes one
    request can be padded with the start of the next without the step
    running longer than the target.
    """

    def __init__(
        self,
        planner: PPChunkPlanner,
        queue_target_ms: float = 0.0,
        num_queued_prefills: int = 0,
    ) -> None:
        self.planner = planner
        self.queue_target_ms = queue_target_ms
        self.num_queued_prefills = num_queued_prefills
        self.target_ms: float | None = None
        self.work_used = 0.0

    def _request_target(self, request: "Request", num_computed_tokens: int) -> float:
        T = request.pp_chunk_target_ms
        if T is None:
            T = self.planner.target_time(request.num_tokens, num_computed_tokens)
        return T

    def _step_target(self, request: "Request", num_computed_tokens: int) -> float:
        return max(
            self._request_target(request, num_computed_tokens),
            self.queue_target_ms,
            self.planner.backlog_target_time(
                self.num_queued_prefills, num_computed_tokens
            ),
        )

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
        target_ms = self.target_ms
        if target_ms is None:
            target_ms = self._step_target(request, num_computed_tokens)
        budget = target_ms - planner.cost.overhead(num_computed_tokens) - self.work_used
        remaining = request.num_tokens - num_computed_tokens
        cap = planner.max_chunk(budget, num_computed_tokens, remaining)
        floor = min(planner.c_min, remaining)
        if self.target_ms is None:
            # The step's first prefill chunk always gets at least c_min.
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
        # The step pays o once, and each prefill chunk its own prefix cost.
        self.work_used += cost.e * num_computed_tokens
        if self.target_ms is None:
            self.target_ms = self._step_target(request, num_computed_tokens)
        if request.pp_chunk_target_ms is None:
            # First chunk: pin the target computed from this prefix length.
            request.pp_chunk_target_ms = self._request_target(
                request, num_computed_tokens
            )


def _is_prefill(request: "Request", num_computed_tokens: int) -> bool:
    # Same predicate as the scheduler's prefill cadence: a request with one
    # token left (a full prefix cache hit, a resumed decode) is a decode.
    return num_computed_tokens < request.num_tokens - 1
