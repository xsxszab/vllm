# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Equal-time prefill chunking for pipeline parallelism.

A prompt split into chunks of durations ``d_1..d_K`` flows through ``S``
pipeline stages as a proportionate flow shop, so its TTFT is
``sum(d_k) + (S - 1) * max(d_k)``. The scheduler therefore sizes every chunk
to the same target duration ``T``, and this module picks ``T`` and converts
it into token counts using the ``pp_dynamic_chunk_cost`` model:

    d(c, L) = o + a * c + b * c * (L + c / 2)

where ``c`` is the chunk size, ``L`` the prefix already computed, ``o`` the
per-chunk fixed overhead, ``a`` the per-token cost and ``b`` the cost per
(query, key) pair of causal attention.
"""

import json
import logging
import math
import statistics
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.request import Request

logger = init_logger(__name__)

# Chunk sizes are rounded down to a multiple of this; measured neutral on
# FlashInfer/TRTLLM attention, but keeps chunks page-aligned for common
# block sizes.
PP_CHUNK_ALIGNMENT = 64
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

    def work(self, c: int, L: int) -> float:
        """Compute-only part of ``d(c, L)``, excluding ``o``."""
        return self.a * c + self.b * c * (L + c / 2)

    def total_work(self, N: int, L0: int) -> float:
        """Compute needed to prefill positions ``[L0, N)``, however chunked."""
        return self.a * (N - L0) + self.b * (N * N - L0 * L0) / 2

    def predict(self, c: int, L: int) -> float:
        return self.o + self.work(c, L)

    @classmethod
    def from_json(cls, text: str) -> "ChunkCostModel":
        try:
            data = json.loads(text)
            model = cls(o=float(data["o"]), a=float(data["a"]), b=float(data["b"]))
        except (ValueError, TypeError, KeyError) as e:
            raise ValueError(
                "pp_dynamic_chunk_cost must be a JSON object with numeric "
                f'"o", "a" and "b" keys, got {text!r}'
            ) from e
        if model.o <= 0 or model.a < 0 or model.b < 0 or model.a + model.b <= 0:
            raise ValueError(
                "pp_dynamic_chunk_cost needs o > 0, a >= 0, b >= 0 and "
                f"a + b > 0, got {text!r}"
            )
        return model


@dataclass
class FitReport:
    fit_error: float
    """Mean absolute percentage error of the fit over all samples."""
    num_samples: int
    o_floored: bool
    """True if ``o`` was clamped to its lower bound."""


def _weighted_nnls(
    phi: np.ndarray, t: np.ndarray, fixed: np.ndarray | None = None
) -> np.ndarray:
    """Relative-error weighted least squares with coefficients clamped at 0.

    Each row is scaled by ``1 / t`` because the timing noise grows with the
    duration. A negative coefficient is fixed at zero and the rest refitted,
    at most three times.
    """
    num_feats = phi.shape[1]
    w = 1.0 / t
    coef = np.zeros(num_feats)
    free = np.ones(num_feats, dtype=bool)
    if fixed is not None:
        free &= ~np.isfinite(fixed)
        coef[~free] = fixed[~free]
    for _ in range(3):
        if not free.any():
            break
        resid = t - phi[:, ~free] @ coef[~free]
        sol, *_ = np.linalg.lstsq(phi[:, free] * w[:, None], resid * w, rcond=None)
        coef[free] = sol
        negative = free & (coef < 0)
        if not negative.any():
            break
        coef[negative] = 0.0
        free &= ~negative
    return np.maximum(coef, 0.0)


def fit_chunk_cost(
    samples: list[tuple[int, int, float]], o_floor_frac: float = 0.01
) -> tuple[ChunkCostModel, FitReport]:
    """Fit ``o, a, b`` to ``(chunk_size, prefix_len, time_ms)`` samples.

    ``o`` is clamped to ``o_floor_frac`` of the median sample time: a fit
    with ``o -> 0`` would drive the target time to zero and squeeze every
    chunk down to the minimum size.

    Raises:
        ValueError: fewer than two distinct chunk sizes or prefix lengths.

    """
    if len({c for c, _, _ in samples}) < 2 or len({L for _, L, _ in samples}) < 2:
        raise ValueError(
            "fitting the chunk cost model needs at least two distinct chunk "
            "sizes and two distinct prefix lengths"
        )
    if any(t <= 0 for _, _, t in samples):
        raise ValueError("chunk cost samples must have positive durations")

    phi = np.array([[1.0, c, c * (L + c / 2)] for c, L, _ in samples])
    t = np.array([t for _, _, t in samples], dtype=float)
    coef = _weighted_nnls(phi, t)

    o_floor = o_floor_frac * statistics.median(t.tolist())
    o_floored = coef[0] < o_floor
    if o_floored:
        fixed = np.array([o_floor, np.nan, np.nan])
        coef = _weighted_nnls(phi, t, fixed=fixed)

    cost = ChunkCostModel(o=float(coef[0]), a=float(coef[1]), b=float(coef[2]))
    pred = phi @ coef
    fit_error = float(np.mean(np.abs(pred - t) / t))
    return cost, FitReport(
        fit_error=fit_error, num_samples=len(samples), o_floored=o_floored
    )


class PPChunkPlanner:
    """Turns the cost model into a per-request target time and chunk sizes.

    Args:
        cost: fitted chunk cost model.
        pp_size: number of pipeline stages ``S``.
        gran: chunk sizes are rounded down to a multiple of this.
        c_min: smallest chunk worth scheduling; keeps GEMMs out of the
            bandwidth-bound regime.
        c_max: largest chunk (the scheduler's token budget).

    """

    def __init__(
        self,
        cost: ChunkCostModel,
        pp_size: int,
        gran: int,
        c_min: int,
        c_max: int,
    ) -> None:
        assert gran >= 1 and c_min >= 1 and c_max >= 1
        self.cost = cost
        self.pp_size = pp_size
        self.gran = gran
        self.c_min = c_min
        self.c_max = c_max

    def target_time(self, N: int, L0: int) -> float:
        """Target duration ``T`` (ms) of every chunk of an ``N``-token prompt.

        ``T* = o + sqrt(W * o / (S - 1))`` minimizes
        ``W + W * o / (T - o) + (S - 1) * T``; it is then clamped so the
        first chunk is at least ``c_min`` and at most ``c_max`` tokens.
        """
        cost = self.cost
        remaining = max(N - L0, 0)
        if remaining == 0:
            return cost.o
        if self.pp_size > 1:
            W = cost.total_work(N, L0)
            T = cost.o + math.sqrt(W * cost.o / (self.pp_size - 1))
        else:
            T = math.inf
        T = max(T, cost.o + cost.work(min(self.c_min, remaining), L0))
        T = min(T, cost.o + cost.work(min(self.c_max, remaining), L0))
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
        aligned = int(c) // self.gran * self.gran
        if aligned == 0 and remaining <= self.gran:
            return remaining
        return aligned

    def predict_ttft(self, N: int, L0: int, T: float) -> float:
        """Closed-form TTFT (ms) of an equal-time split at target ``T``."""
        W = self.cost.total_work(N, L0)
        if W <= 0:
            return self.cost.o * self.pp_size
        return W + W * self.cost.o / (T - self.cost.o) + (self.pp_size - 1) * T

    def queue_target_time(
        self,
        queue: Iterable["Request"],
        computed_tokens: Callable[["Request"], int] | None = None,
    ) -> float:
        """Step target (ms) minimizing the mean TTFT of the queued prefills.

        Under FCFS with equal-time steps, the ``j``-th queued prefill gets its
        first token after ``W_j + W_j * o / (T - o) + (S - 1) * T``, where
        ``W_j`` is its remaining work plus that of every prefill ahead of it.
        The mean TTFT is therefore minimized by ``target_time``'s formula with
        ``W`` replaced by the mean of ``W_j``, which grows with the backlog and
        so trades the pipeline fill of the head request for throughput.

        ``computed_tokens`` gives the prefix each request starts from, so that
        waiting requests can be credited with their expected prefix-cache
        hits; it defaults to ``num_computed_tokens``. The scan stops once the
        head request's chunk alone would fill the token budget. Returns 0 with
        fewer than two queued prefills, leaving a lone request at its own
        target.
        """
        if self.pp_size <= 1:
            return 0.0
        cost = self.cost
        num_reqs = 0
        cum_work = 0.0
        sum_cum_work = 0.0
        full_step_work = math.inf
        for request in queue:
            L = (
                computed_tokens(request)
                if computed_tokens is not None
                else request.num_computed_tokens
            )
            if not _is_prefill(request, L):
                continue
            N = request.num_tokens
            if num_reqs == 0:
                full_step_work = cost.work(self.c_max, L)
            cum_work += cost.total_work(N, L)
            sum_cum_work += cum_work
            num_reqs += 1
            if (
                num_reqs == _MAX_QUEUE_SCAN
                or sum_cum_work / num_reqs * cost.o / (self.pp_size - 1)
                >= full_step_work**2
            ):
                break
        if num_reqs < 2:
            return 0.0
        return cost.o + math.sqrt(sum_cum_work / num_reqs * cost.o / (self.pp_size - 1))

    def plan(self, N: int, L0: int, T: float | None = None) -> list[int]:
        """Chunk sizes a lone request would get, in scheduling order."""
        if T is None:
            T = self.target_time(N, L0)
        chunks: list[int] = []
        L = L0
        while L < N:
            remaining = N - L
            c = self.max_chunk(T - self.cost.o, L, remaining)
            c = max(c, min(self.c_min, remaining))
            chunks.append(c)
            L += c
        return chunks


class PPChunkStep:
    """Caps the prefill chunks of one scheduling step to a shared target.

    The first prefill chunk scheduled in a step sets the step's time target:
    the larger of its request's own ``T`` and ``queue_target_ms``. Later
    requests only get what is left of it, so a chunk that finishes one
    request can be padded with the start of the next without the step
    running longer than the target.
    """

    def __init__(self, planner: PPChunkPlanner, queue_target_ms: float = 0.0) -> None:
        self.planner = planner
        self.queue_target_ms = queue_target_ms
        self.target_ms: float | None = None
        self.work_used = 0.0

    def _request_target(self, request: "Request", num_computed_tokens: int) -> float:
        T = request.pp_chunk_target_ms
        if T is None:
            T = self.planner.target_time(request.num_tokens, num_computed_tokens)
        return T

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
            target_ms = max(
                self._request_target(request, num_computed_tokens),
                self.queue_target_ms,
            )
        budget = target_ms - planner.cost.o - self.work_used
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
        planner = self.planner
        self.work_used += planner.cost.work(num_new_tokens, num_computed_tokens)
        if not _is_prefill(request, num_computed_tokens):
            return
        T = self._request_target(request, num_computed_tokens)
        if self.target_ms is None:
            self.target_ms = max(T, self.queue_target_ms)
        if request.pp_chunk_target_ms is None:
            # First chunk: pin the target computed from this prefix length.
            request.pp_chunk_target_ms = T
            if logger.isEnabledFor(logging.DEBUG):
                N = request.num_tokens
                logger.debug(
                    "[pp-dyn-chunk] req=%s N=%d L0=%d T=%.2fms est_chunks=%d "
                    "est_ttft=%.3fs",
                    request.request_id,
                    N,
                    num_computed_tokens,
                    T,
                    len(planner.plan(N, num_computed_tokens, T)),
                    planner.predict_ttft(N, num_computed_tokens, T) / 1000,
                )


def _is_prefill(request: "Request", num_computed_tokens: int) -> bool:
    # Same predicate as the scheduler's prefill cadence: a request with one
    # token left (a full prefix cache hit, a resumed decode) is a decode.
    return num_computed_tokens < request.num_tokens - 1
