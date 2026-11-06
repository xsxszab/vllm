# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pipeline Parallelism utils for V2 Model Runner."""

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

import vllm.envs as envs
from vllm.distributed.parallel_state import get_pp_group
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.worker.gpu.input_batch import InputBatch


@dataclass
class PendingRecv:
    """Per-step slot data for a deferred postprocess on the main stream."""

    event: torch.cuda.Event

    sampled_tokens: torch.Tensor  # [num_reqs, sample_len]
    num_sampled: torch.Tensor  # [num_reqs]
    num_rejected: torch.Tensor  # [num_reqs]
    idx_mapping: torch.Tensor  # [num_reqs]
    idx_mapping_np: np.ndarray  # [num_reqs]
    # Records which rows need a deferred postprocess (bool).
    need_sampled_mask: np.ndarray  # [num_reqs]
    # Snapshot of slot generation counters at receive time, used to
    # detect requests aborted since then.
    gen_at_receive_np: np.ndarray  # [num_reqs]
    draft_tokens: torch.Tensor | None = None  # [num_reqs, num_speculative_steps]


@dataclass
class PendingConfidences:
    """One step's draft confidences, applied ring_depth steps later on every rank."""

    # Broadcast receive event on non-last ranks; None on the last rank.
    event: torch.cuda.Event | None
    confidences: torch.Tensor  # [num_reqs, num_speculative_steps], float32
    idx_mapping_np: np.ndarray  # [num_reqs]
    gen_np: np.ndarray  # [num_reqs]


def compute_need_sampled_mask(input_batch: InputBatch) -> np.ndarray | None:
    """Return a bool array of shape `[input_batch.num_reqs]` marking requests
    with outputs that might be needed in a subsequent (decode) step.
    Returns None if no sampled outputs are needed in the requests' next step."""
    old_computed = input_batch.num_computed_tokens_np
    prefill_len = input_batch.prefill_len_np
    # Exclude non-final prefill chunks (they don't produce a sample).
    need_sampled_mask = old_computed + input_batch.num_scheduled_tokens >= prefill_len
    if input_batch.max_seq_len_np is not None:
        # Also exclude final prefill chunks whose single sampled token reaches
        # the request's length cap.
        finished_prefill = prefill_len + 1 >= input_batch.max_seq_len_np
        finished_prefill &= input_batch.is_prefilling_np
        need_sampled_mask &= ~finished_prefill
    return need_sampled_mask if need_sampled_mask.any() else None


class PPHandler:
    """Runs the PP sampled-token broadcast/recv on a side stream so the
    default stream isn't gated by the matching peer call. Step T's recv is
    consumed at step T+pp_size via `get_prev_sampled_outputs`.

    Uses a dedicated NCCL communicator (sibling of the PP `device_group`)
    for the broadcast so it does not serialize on the wire with the
    inter-stage hidden-state p2p send/recv ops.
    """

    def __init__(
        self, max_num_reqs: int, num_speculative_steps: int, device: torch.device
    ):
        self.is_last_rank = get_pp_group().is_last_rank
        self.last_rank = get_pp_group().last_rank
        self.max_sample_len = num_speculative_steps + 1
        self.num_speculative_steps = num_speculative_steps
        self.device = device
        self.main_stream = torch.cuda.current_stream(device)
        self.broadcast_stream = torch.cuda.Stream(device)

        # On non-last ranks, a FIFO with one entry per in-flight step: the entry
        # pushed by step T's `receive` is consumed pp_size steps later. Pre-seeded
        # with pp_size None placeholders so the first pp_size consumes are no-ops.
        # None means no postprocess is pending for that step (broadcast skipped).
        # Only XPU can disable microbatching via VLLM_XPU_PP_MICROBATCH.
        ring_depth = get_pp_group().world_size
        if current_platform.is_xpu() and not envs.VLLM_XPU_PP_MICROBATCH:
            ring_depth = 1
        self.ring_depth = ring_depth
        self.queue: deque[PendingRecv | None] = (
            deque() if self.is_last_rank else deque([None] * ring_depth)
        )
        # Adaptive verification: the same FIFO for draft confidences, but kept on
        # every rank, the last included, so that all ranks fold in a step's
        # confidences at the same later step. See `pop_confidences`.
        self.confidence_queue: deque[PendingConfidences | None] | None = None

        # Per req-index generation counter, incremented every time a request
        # index is freed in RequestStats. Used for invalidating freed req data
        # between PP decodes.
        self.req_idx_gen_np = np.zeros(max_num_reqs, dtype=np.int32)

        # Dedicated subgroup for the sampled-token broadcast.
        self.broadcast_group = get_pp_group().make_sibling_device_group(
            group_desc="pp_broadcast"
        )
        # Whether sampled tokens were broadcast this step.
        self.sampled_tokens_broadcast = False
        self.aux_hidden_state_relay_keys: tuple[str, ...] = ()

    def on_req_idx_freed(self, req_idx: int) -> None:
        self.req_idx_gen_np[req_idx] += 1

    def _sample_len(self, input_batch: InputBatch) -> int:
        # Width of the sampled-token broadcast. Must be the same on all ranks.
        return self.max_sample_len if input_batch.num_draft_tokens > 0 else 1

    def configure_aux_hidden_state_relay(self, model: torch.nn.Module) -> None:
        from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
            aux_hidden_state_relay_keys,
        )

        self.aux_hidden_state_relay_keys = aux_hidden_state_relay_keys(model)

    def relay_aux_hidden_states(
        self,
        intermediate_tensors: IntermediateTensors | None,
        output_intermediate_tensors: IntermediateTensors,
    ) -> IntermediateTensors:
        if not self.aux_hidden_state_relay_keys:
            return output_intermediate_tensors
        assert intermediate_tensors is not None
        return IntermediateTensors(
            output_intermediate_tensors.tensors
            | {
                key: intermediate_tensors[key]
                for key in self.aux_hidden_state_relay_keys
            }
        )

    def get_prev_sampled_outputs(self) -> dict[str, torch.Tensor | None] | None:
        """Consume the entry from pp_size steps ago and wait for its recv event,
        then filter out entries whose request was freed since `receive`.
        """
        if not self.queue:
            return None
        slot = self.queue.popleft()
        # Reserve this step's slot; `receive` overwrites it if applicable.
        self.queue.append(None)
        if slot is None:
            return None

        # Skip requests which did not need sampled output and/or those already
        # finished. The post_update kernel skips the -1 entries.
        freed = self.req_idx_gen_np[slot.idx_mapping_np] != slot.gen_at_receive_np
        exclude_mask = freed | ~slot.need_sampled_mask
        idx_mapping = slot.idx_mapping
        if exclude_mask.any():
            if exclude_mask.all():
                # No states require update anymore.
                return None
            # Filter excluded request indices.
            idx_mapping_np = np.where(exclude_mask, -1, slot.idx_mapping_np)
            idx_mapping = async_tensor_h2d(idx_mapping_np, device=self.device)

        self.main_stream.wait_event(slot.event)
        return dict(
            sampled_tokens=slot.sampled_tokens,
            num_sampled=slot.num_sampled,
            num_rejected=slot.num_rejected,
            idx_mapping=idx_mapping,
            draft_tokens=slot.draft_tokens,
        )

    def enable_confidence_relay(self) -> None:
        self.confidence_queue = deque([None] * self.ring_depth)

    def pop_confidences(self) -> tuple[torch.Tensor, np.ndarray, np.ndarray] | None:
        """Consume the confidences pushed ring_depth steps ago. Every rank calls
        this once per executed step, so all ranks apply the same updates at the
        same step. Returns (confidences, live rows, their request indices)."""
        assert self.confidence_queue is not None
        slot = self.confidence_queue.popleft()
        # Reserve this step's slot; broadcast_drafts/receive fill it.
        self.confidence_queue.append(None)
        if slot is None:
            return None
        live = self.req_idx_gen_np[slot.idx_mapping_np] == slot.gen_np
        if not live.any():
            return None
        if slot.event is not None:
            self.main_stream.wait_event(slot.event)
        rows = np.flatnonzero(live)
        return slot.confidences, rows, slot.idx_mapping_np[rows]

    def broadcast_drafts(
        self,
        draft_tokens: torch.Tensor,
        input_batch: InputBatch | None = None,
        confidences: torch.Tensor | None = None,
    ) -> None:
        """Broadcast draft proposals so non-last ranks can embed real token ids,
        and, with adaptive verification, the drafts' confidences."""
        assert self.is_last_rank
        if self.sampled_tokens_broadcast:
            if confidences is not None:
                assert input_batch is not None and self.confidence_queue is not None
                # The speculator reuses its buffer next step.
                confidences = confidences[: input_batch.num_reqs].clone()
                self.confidence_queue[-1] = PendingConfidences(
                    None,
                    confidences,
                    input_batch.idx_mapping_np,
                    self.req_idx_gen_np[input_batch.idx_mapping_np],
                )
            with torch.cuda.stream(self.broadcast_stream):
                self.broadcast_stream.wait_stream(self.main_stream)
                torch.distributed.broadcast(
                    draft_tokens, src=self.last_rank, group=self.broadcast_group
                )
                draft_tokens.record_stream(self.broadcast_stream)
                if confidences is not None:
                    torch.distributed.broadcast(
                        confidences, src=self.last_rank, group=self.broadcast_group
                    )
                    confidences.record_stream(self.broadcast_stream)

    def receive(
        self, input_batch: InputBatch, recv_drafts: bool, recv_confidences: bool = False
    ) -> bool:
        """Returns True iff sampled tokens need to be gathered from *all*
        requests in the batch. `recv_drafts` and `recv_confidences` must match
        whether the last rank calls `broadcast_drafts` for this step, and with
        confidences."""
        assert not self.is_last_rank
        need_sampled_mask = compute_need_sampled_mask(input_batch)
        if need_sampled_mask is None:
            # Leave this step's reserved slot as None.
            return False
        recv_confidences = recv_drafts and recv_confidences

        # Snapshot the per-slot generation counter so a later free of any of
        # these RequestStates request indices is detectable at consume time.
        gen_at_receive_np = self.req_idx_gen_np[input_batch.idx_mapping_np]

        num_reqs = input_batch.num_reqs
        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            sample_len = self._sample_len(input_batch)
            sampled_tokens = torch.empty(
                num_reqs, sample_len, dtype=torch.int64, device=self.device
            )
            combined = torch.empty(2, num_reqs, dtype=torch.int32, device=self.device)
            torch.distributed.broadcast(
                sampled_tokens, src=self.last_rank, group=self.broadcast_group
            )
            torch.distributed.broadcast(
                combined, src=self.last_rank, group=self.broadcast_group
            )
            draft_tokens = None
            if recv_drafts:
                draft_tokens = sampled_tokens.new_empty(
                    num_reqs, self.num_speculative_steps
                )
                torch.distributed.broadcast(
                    draft_tokens, src=self.last_rank, group=self.broadcast_group
                )
            confidences = None
            if recv_confidences:
                confidences = torch.empty(
                    num_reqs,
                    self.num_speculative_steps,
                    dtype=torch.float32,
                    device=self.device,
                )
                torch.distributed.broadcast(
                    confidences, src=self.last_rank, group=self.broadcast_group
                )
            event = self.broadcast_stream.record_event()
            num_sampled, num_rejected = combined.unbind(dim=0)
            # Must record_stream since these were allocated on broadcast stream but
            # later used on the main stream.
            sampled_tokens.record_stream(self.main_stream)
            combined.record_stream(self.main_stream)
            if draft_tokens is not None:
                draft_tokens.record_stream(self.main_stream)
            if confidences is not None:
                confidences.record_stream(self.main_stream)
        if confidences is not None:
            assert self.confidence_queue is not None
            self.confidence_queue[-1] = PendingConfidences(
                event, confidences, input_batch.idx_mapping_np, gen_at_receive_np
            )
        self.queue[-1] = PendingRecv(
            event,
            sampled_tokens,
            num_sampled,
            num_rejected,
            input_batch.idx_mapping,
            input_batch.idx_mapping_np,
            need_sampled_mask,
            gen_at_receive_np,
            draft_tokens,
        )
        return bool(need_sampled_mask.all())

    def broadcast(
        self,
        sampled_token_ids: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        input_batch: InputBatch,
    ) -> None:
        assert self.is_last_rank
        if compute_need_sampled_mask(input_batch) is None:
            # No request needs sampled outputs for a subsequent decode step.
            self.sampled_tokens_broadcast = False
            return

        assert sampled_token_ids.dtype == torch.int64
        self.sampled_tokens_broadcast = True

        if current_platform.is_xpu():
            self.main_stream.synchronize()

        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            # receive() allocates the same width on the other ranks.
            assert sampled_token_ids.shape[-1] == self._sample_len(input_batch)
            torch.distributed.broadcast(
                sampled_token_ids.contiguous(),
                src=self.last_rank,
                group=self.broadcast_group,
            )
            combined = torch.stack((num_sampled, num_rejected), dim=0)
            torch.distributed.broadcast(
                combined, src=self.last_rank, group=self.broadcast_group
            )
            for tensor in (sampled_token_ids, num_sampled, num_rejected):
                tensor.record_stream(self.broadcast_stream)
