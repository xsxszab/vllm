# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup profiling of the ``pp_dynamic_chunking`` cost model.

Every pipeline stage times its own forward pass on synthetic chunked prefills
over a grid of chunk sizes and prefix lengths, without the inter-stage
send/recv. The engine fits ``d(c, L) = o + a * c + b * c * (L + c / 2)`` to
the slowest stage at each grid point and hands it to the scheduler.
"""

import json
import statistics
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.v1.core.sched.pp_chunk_planner import ChunkCostModel

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

_PREFIX_FRACTIONS = (0.0, 0.25, 0.5, 1.0)
_NUM_WARMUP = 1
_NUM_TIMED = 3
# An o near zero would drive the target time to zero and squeeze every chunk
# down to c_min, so o is kept at least this fraction of the median sample.
_O_FLOOR_FRAC = 0.01


def _grid(c_min: int, c_max: int, max_prefix: int) -> list[tuple[int, int]]:
    chunks = sorted(
        {min(c_min << i, c_max) for i in range(32) if c_min << i < 2 * c_max}
    )
    prefixes = sorted({int(f * max_prefix) // 64 * 64 for f in _PREFIX_FRACTIONS})
    return [(c, L) for L in prefixes for c in chunks]


def _nonneg_lstsq(phi: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Least squares on relative error, dropping negative coefficients."""
    w = 1.0 / t
    coef = np.zeros(phi.shape[1])
    free = np.ones(phi.shape[1], dtype=bool)
    while free.any():
        sol, *_ = np.linalg.lstsq(phi[:, free] * w[:, None], t * w, rcond=None)
        coef[:] = 0.0
        coef[free] = sol
        if (sol >= 0).all():
            break
        free[free] = sol >= 0
    return coef


def fit_chunk_cost(
    samples: list[tuple[int, int, float]],
) -> tuple[ChunkCostModel, float]:
    """Fit the cost model to ``(chunk_size, prefix_len, time_ms)`` samples.

    Returns:
        The model and its mean relative error over the samples.

    Raises:
        ValueError: fewer than two distinct chunk sizes or prefix lengths.

    """
    if len({c for c, _, _ in samples}) < 2 or len({L for _, L, _ in samples}) < 2:
        raise ValueError(
            "fitting the chunk cost model needs at least two distinct chunk "
            "sizes and two distinct prefix lengths"
        )
    phi = np.array([[1.0, c, c * (L + c / 2)] for c, L, _ in samples])
    t = np.array([t for _, _, t in samples], dtype=float)
    coef = _nonneg_lstsq(phi, t)
    o_floor = _O_FLOOR_FRAC * float(np.median(t))
    if coef[0] < o_floor:
        coef = np.concatenate(([o_floor], _nonneg_lstsq(phi[:, 1:], t - o_floor)))
    fit_error = float(np.mean(np.abs(phi @ coef - t) / t))
    o, a, b = (float(x) for x in coef)
    return ChunkCostModel(o=o, a=a, b=b), fit_error


@torch.inference_mode()
def _profile_on_worker(
    worker: Any, grid: list[tuple[int, int]]
) -> tuple[int, list[float]]:
    """Median forward time (ms) of this stage at each (chunk, prefix) point."""
    from vllm.benchmarks.forward.batch import build_batch
    from vllm.benchmarks.forward.kv_init import zero_kv_blocks
    from vllm.benchmarks.forward.shapes import BenchCase, BenchRequest
    from vllm.distributed.parallel_state import get_pp_group

    runner = worker.model_runner
    kv_config = runner.kv_cache_config
    context = runner.vllm_config.compilation_config.static_forward_context
    kv_caches = {
        name: getattr(layer, "kv_cache", None) for name, layer in context.items()
    }
    attn_groups = [group for groups in runner.attn_groups for group in groups]
    rng = np.random.default_rng(0)
    times_ms = []
    for c, L in grid:
        request = BenchRequest(query_len=c, kv_len=L + c, chunked=L > 0)
        batch = build_batch(
            BenchCase(f"pp-chunk-{c}@{L}", (request,)),
            kv_config.kv_cache_groups,
            kv_config.num_blocks,
            runner.model_config.get_vocab_size(),
            rng,
            runner.parallel_config.decode_context_parallel_size,
        )
        zero_kv_blocks(kv_caches, attn_groups, batch, kv_config.num_blocks)
        samples = []
        for i in range(_NUM_WARMUP + _NUM_TIMED):
            torch.accelerator.synchronize()
            start = time.perf_counter()
            runner.execute_model(
                batch.scheduler_output, intermediate_tensors=runner.intermediate_tensors
            )
            torch.accelerator.synchronize()
            if i >= _NUM_WARMUP:
                samples.append((time.perf_counter() - start) * 1000)
            # No sample_tokens follows; remove the request for the next run.
            runner.execute_model_state = None
            runner.execute_model(batch.finish())
        times_ms.append(statistics.median(samples))
    return get_pp_group().rank_in_group, times_ms


def profile_pp_chunk_cost(
    vllm_config: "VllmConfig", collective_rpc: Callable[..., list[Any]]
) -> None:
    """Profile the chunk cost model if ``pp_dynamic_chunking`` needs one.

    Call after the KV cache is allocated and before the scheduler is built,
    while no request holds KV blocks. On failure, dynamic chunking is
    disabled rather than failing startup.
    """
    scheduler_config = vllm_config.scheduler_config
    if (
        not scheduler_config.pp_dynamic_chunking
        or scheduler_config.pp_dynamic_chunk_cost is not None
    ):
        return
    c_max = scheduler_config.max_num_batched_tokens
    c_min = min(scheduler_config.pp_dynamic_chunk_min_tokens, c_max)
    # Half the KV cache leaves room for models with several KV cache groups.
    kv_tokens = (vllm_config.cache_config.num_gpu_blocks or 0) * (
        vllm_config.cache_config.block_size
    )
    max_prefix = min(
        vllm_config.model_config.max_model_len - c_max, kv_tokens // 2 - c_max
    )
    grid = _grid(c_min, c_max, max(max_prefix, 0))
    try:
        reports = collective_rpc(_profile_on_worker, args=(grid,))
        # The slowest stage paces the pipeline.
        stage_ms = [max(times[i] for _, times in reports) for i in range(len(grid))]
        cost, fit_error = fit_chunk_cost(
            [(c, L, t) for (c, L), t in zip(grid, stage_ms)]
        )
    except Exception:
        logger.exception(
            "pp_dynamic_chunking: profiling the chunk cost model failed; "
            "disabling dynamic chunking."
        )
        scheduler_config.pp_dynamic_chunking = False
        return
    scheduler_config.pp_dynamic_chunk_cost = json.dumps(
        {"o": cost.o, "a": cost.a, "b": cost.b}
    )
    logger.info(
        "pp_dynamic_chunking: profiled %d points on %d stages, cost model %s "
        "(mean fit error %.1f%%). Pass it as --pp-dynamic-chunk-cost to skip "
        "profiling.",
        len(grid),
        len({rank for rank, _ in reports}),
        scheduler_config.pp_dynamic_chunk_cost,
        100 * fit_error,
    )
