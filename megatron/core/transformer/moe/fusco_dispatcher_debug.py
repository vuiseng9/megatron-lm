"""Debug version of fusco_dispatcher with extensive bounds checking."""
import logging
from abc import ABC, abstractmethod
import os
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist

from megatron.core import utils
from megatron.core.config import is_experimental_enabled
from megatron.core.fusions.fused_indices_converter import fused_indices_to_multihot
from megatron.core.fusions.fused_pad_routing_map import fused_pad_routing_map
from megatron.core.jit import jit_fuser
from megatron.core.tensor_parallel import (
    all_to_all,
    gather_from_sequence_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
)
from megatron.core.transformer.enums import CudaGraphScope
from megatron.core.transformer.moe.fused_a2a import (
    fused_combine,
    fused_dispatch,
    hybrid_ep_combine,
    hybrid_ep_dispatch,
    set_deepep_num_sms,
)
from megatron.core.transformer.moe.moe_utils import (
    ProcessGroupCollection,
    get_align_size_for_quantization,
    get_capacity,
    maybe_move_tensor_to_cpu,
    pad_routing_map,
    permute,
    sort_chunks_by_idxs,
    unpermute,
)
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.moe.token_dispatcher import MoETokenDispatcher

from fusco import FUSCO
import idxtools

GLOBAL_FUSCO = None


def debug_print(rank, msg):
    """Print debug message with rank prefix."""
    if rank == 0 or os.environ.get("DEBUG_ALL_RANKS", "0") == "1":
        print(f"[RANK {rank}] {msg}", flush=True)


def check_indices_bounds(indices, max_val, name, rank):
    """Check if indices are within valid bounds."""
    if indices.numel() == 0:
        return True
    min_idx = indices.min().item()
    max_idx = indices.max().item()
    if min_idx < 0 or max_idx >= max_val:
        print(f"[RANK {rank}] ERROR: {name} out of bounds! "
              f"range=[{min_idx}, {max_idx}], valid=[0, {max_val-1}]", flush=True)
        return False
    return True


def gather_along_first_dim(input_, group):
    world_size = torch.distributed.get_world_size(group=group)

    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] * world_size

    output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
    torch.distributed.all_gather_into_tensor(output, input_.contiguous(), group=group)

    return output


class MoEFuscoTokenDispatcher(MoETokenDispatcher):
    def __init__(
        self,
        num_local_experts: int,
        local_expert_indices: List[int],
        config: TransformerConfig,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        super().__init__(config=config, pg_collection=pg_collection)

        self.num_local_experts = num_local_experts
        self.local_expert_indices = local_expert_indices
        self.num_experts = config.num_moe_experts * self.tp_size
        self.topk = config.moe_router_topk * self.tp_size
        assert self.tp_size * self.ep_size > 1, "Flex token dispatcher requires TPxEP > 1"

        self.rank = dist.get_rank()
        debug_print(self.rank, f"=== FUSCO DISPATCHER INIT ===")
        debug_print(self.rank, f"  num_local_experts={num_local_experts}")
        debug_print(self.rank, f"  local_expert_indices={local_expert_indices}")
        debug_print(self.rank, f"  num_experts (total)={self.num_experts}")
        debug_print(self.rank, f"  topk={self.topk}")
        debug_print(self.rank, f"  tp_size={self.tp_size}, ep_size={self.ep_size}")
        debug_print(self.rank, f"  tp_ep world_size={self.tp_size * self.ep_size}")

        global GLOBAL_FUSCO
        if GLOBAL_FUSCO is None:
            self.fusco_pg = self.tp_ep_group if self.tp_size > 1 else self.ep_group
            GLOBAL_FUSCO = FUSCO(
                group_ranks=[dist.get_process_group_ranks(self.fusco_pg)],
                library_path="/workspace/fusco-ep/lib/libfusco.so",
            )
        self.fusco = GLOBAL_FUSCO
        
    def dispatch_preprocess(self, tokens, routing_map, probs):
        self.hidden_shape = tokens.shape
        
        debug_print(self.rank, f"=== DISPATCH PREPROCESS ===")
        debug_print(self.rank, f"  tokens.shape={tokens.shape}")
        debug_print(self.rank, f"  routing_map.shape={routing_map.shape}")
        debug_print(self.rank, f"  probs.shape={probs.shape}")
        
        if self.tp_size > 1:
            num_tokens = routing_map.shape[0]
            probs = (
                probs.reshape(num_tokens, self.ep_size, 1, self.num_local_experts)
                        .expand(-1, -1, self.tp_size, -1)
                            .reshape(num_tokens, self.num_experts)
                ).contiguous()
            debug_print(self.rank, f"  probs after TP expand: {probs.shape}")

        k_probs, k_indices = torch.topk(probs, self.topk, dim=-1, largest=True, sorted=False)
        k_indices = k_indices.to(torch.int64)

        debug_print(self.rank, f"  k_probs.shape={k_probs.shape}")
        debug_print(self.rank, f"  k_indices.shape={k_indices.shape}")
        debug_print(self.rank, f"  k_indices min={k_indices.min().item()}, max={k_indices.max().item()}")
        
        # CRITICAL CHECK: k_indices should be in [0, num_experts)
        check_indices_bounds(k_indices, self.num_experts, "k_indices", self.rank)

        self.probs = k_probs
        self.indices_shape = k_indices.shape

        num_local_tokens_per_expert = torch.bincount(k_indices.view(-1), minlength=self.num_experts)
        debug_print(self.rank, f"  num_local_tokens_per_expert.shape={num_local_tokens_per_expert.shape}")
        debug_print(self.rank, f"  num_local_tokens_per_expert.sum()={num_local_tokens_per_expert.sum().item()}")

        num_local_tokens_per_rank = num_local_tokens_per_expert.view(
            self.ep_size * self.tp_size, self.num_local_experts
        ).sum(dim=1)
        debug_print(self.rank, f"  num_local_tokens_per_rank.shape={num_local_tokens_per_rank.shape}")
        debug_print(self.rank, f"  num_local_tokens_per_rank={num_local_tokens_per_rank.tolist()}")

        topk = k_indices.size(1)
        flatten_indices = k_indices.view(-1)
        num_flatten = flatten_indices.numel()
        debug_print(self.rank, f"  flatten_indices.numel()={num_flatten}")

        self.sendindices_unique = torch.argsort(flatten_indices, stable=True).contiguous()
        self.sendindices_with_duplicates = (self.sendindices_unique // topk).contiguous()
        self.send_splits = num_local_tokens_per_rank.to(torch.device("cpu"))

        debug_print(self.rank, f"  sendindices_unique.shape={self.sendindices_unique.shape}")
        debug_print(self.rank, f"  sendindices_unique min={self.sendindices_unique.min().item()}, max={self.sendindices_unique.max().item()}")
        debug_print(self.rank, f"  send_splits={self.send_splits.tolist()}")
        
        # CRITICAL CHECK: sendindices should be in [0, num_tokens * topk)
        check_indices_bounds(self.sendindices_unique, num_flatten, "sendindices_unique", self.rank)

        num_global_tokens_per_expert = gather_along_first_dim(
            num_local_tokens_per_expert, self.fusco_pg
        ).reshape(self.ep_size * self.tp_size, self.num_experts)
        debug_print(self.rank, f"  num_global_tokens_per_expert.shape={num_global_tokens_per_expert.shape}")

        num_global_tokens_per_local_expert = num_global_tokens_per_expert[
            :, self.local_expert_indices[0] : self.local_expert_indices[-1] + 1
        ].contiguous()
        debug_print(self.rank, f"  num_global_tokens_per_local_expert.shape={num_global_tokens_per_local_expert.shape}")
        debug_print(self.rank, f"  local_expert_indices range: [{self.local_expert_indices[0]}, {self.local_expert_indices[-1]}]")

        num_tokens_per_local_expert = num_global_tokens_per_local_expert.sum(dim=0)
        num_tokens_per_ep = num_global_tokens_per_local_expert.sum(dim=1)

        debug_print(self.rank, f"  num_tokens_per_local_expert.shape={num_tokens_per_local_expert.shape}")
        debug_print(self.rank, f"  num_tokens_per_local_expert={num_tokens_per_local_expert.tolist()}")
        debug_print(self.rank, f"  num_tokens_per_ep.shape={num_tokens_per_ep.shape}")
        debug_print(self.rank, f"  num_tokens_per_ep={num_tokens_per_ep.tolist()}")

        self.num_ep_tokens = num_global_tokens_per_local_expert.sum()
        debug_print(self.rank, f"  num_ep_tokens (total recv)={self.num_ep_tokens.item()}")

        if self.num_ep_tokens > 0:
            # Synchronize before calling CUDA kernel to ensure inputs are ready
            torch.cuda.synchronize()
            self.recvindices = idxtools.indices_gen(
                num_global_tokens_per_local_expert,
                num_tokens_per_local_expert,
                num_tokens_per_ep,
                self.num_ep_tokens.item(),
            )
            # Synchronize after CUDA kernel to catch any errors
            torch.cuda.synchronize()
        else:
            self.recvindices = torch.empty(0, dtype=torch.int64, device=torch.cuda.current_device())
        
        debug_print(self.rank, f"  recvindices.shape={self.recvindices.shape}")
        if self.recvindices.numel() > 0:
            debug_print(self.rank, f"  recvindices min={self.recvindices.min().item()}, max={self.recvindices.max().item()}")
            # CRITICAL CHECK: recvindices should be in [0, num_ep_tokens)
            if not check_indices_bounds(self.recvindices, self.num_ep_tokens.item(), "recvindices", self.rank):
                print(f"[RANK {self.rank}] FATAL: recvindices out of bounds!", flush=True)
                print(f"[RANK {self.rank}]   num_global_tokens_per_local_expert:\n{num_global_tokens_per_local_expert}", flush=True)
                print(f"[RANK {self.rank}]   num_tokens_per_local_expert: {num_tokens_per_local_expert.tolist()}", flush=True)
                print(f"[RANK {self.rank}]   num_tokens_per_ep: {num_tokens_per_ep.tolist()}", flush=True)
            
            # Check for uniqueness of recvindices (they should be a permutation)
            unique_count = self.recvindices.unique().numel()
            if unique_count != self.recvindices.numel():
                print(f"[RANK {self.rank}] WARNING: recvindices not unique! "
                      f"total={self.recvindices.numel()}, unique={unique_count}", flush=True)
        
        self.recv_splits = num_tokens_per_ep.to(torch.device("cpu"))
        debug_print(self.rank, f"  recv_splits={self.recv_splits.tolist()}")
        
        # Validate splits sum
        send_sum = self.send_splits.sum().item()
        recv_sum = self.recv_splits.sum().item()
        debug_print(self.rank, f"  VALIDATION: send_splits.sum()={send_sum}, recv_splits.sum()={recv_sum}")
        debug_print(self.rank, f"  VALIDATION: sendindices.numel()={self.sendindices_unique.numel()}, recvindices.numel()={self.recvindices.numel()}")

        self.tokens_per_expert = num_tokens_per_local_expert

        return tokens.view(-1, tokens.shape[-1]), k_probs
    
    def token_dispatch(self, hidden_states, _unused_probs):
        debug_print(self.rank, f"=== TOKEN DISPATCH ===")
        debug_print(self.rank, f"  hidden_states.shape={hidden_states.shape}")
        
        num_tokens = hidden_states.shape[0]
        hidden_dim = hidden_states.shape[-1]
        
        hidden_states_expanded = hidden_states.unsqueeze(1).expand(
            num_tokens, self.topk, -1
        ).reshape(num_tokens * self.topk, -1)
        
        debug_print(self.rank, f"  hidden_states_expanded.shape={hidden_states_expanded.shape}")
        debug_print(self.rank, f"  output_shape will be: ({self.num_ep_tokens.item()}, {hidden_dim})")
        
        # CRITICAL VALIDATION before FUSCO call
        expected_send = self.sendindices_unique.numel()
        actual_input = hidden_states_expanded.shape[0]
        debug_print(self.rank, f"  VALIDATION: input_tokens={actual_input}, sendindices.numel()={expected_send}")
        
        if expected_send > actual_input:
            print(f"[RANK {self.rank}] ERROR: sendindices references beyond input! "
                  f"max_sendindex={self.sendindices_unique.max().item()}, input_size={actual_input}", flush=True)
        
        tokens_by_expert = _FuscoAllToAll.apply(
            self.fusco, 
            hidden_states_expanded,
            (self.num_ep_tokens, hidden_states.shape[-1]),
            self.recvindices, 
            self.sendindices_unique,
            self.recv_splits, 
            self.send_splits)
        
        debug_print(self.rank, f"  tokens_by_expert.shape={tokens_by_expert.shape}")
        return tokens_by_expert, _unused_probs
    
    def dispatch_postprocess(self, dispatched_tokens, _unused_probs):
        debug_print(self.rank, f"=== DISPATCH POSTPROCESS ===")
        debug_print(self.rank, f"  dispatched_tokens.shape={dispatched_tokens.shape}")
        debug_print(self.rank, f"  tokens_per_expert={self.tokens_per_expert.tolist()}")
        
        dummy_probs = torch.ones(dispatched_tokens.shape[0],
                                 dtype=dispatched_tokens.dtype,
                                 device=dispatched_tokens.device)
        return dispatched_tokens, self.tokens_per_expert, dummy_probs
    
    def combine_preprocess(self, expert_outputs):
        debug_print(self.rank, f"=== COMBINE PREPROCESS ===")
        debug_print(self.rank, f"  expert_outputs.shape={expert_outputs.shape}")
        return expert_outputs

    def token_combine(self, expert_outputs):
        debug_print(self.rank, f"=== TOKEN COMBINE ===")
        debug_print(self.rank, f"  expert_outputs.shape={expert_outputs.shape}")
        
        output_shape = (self.indices_shape[0] * self.indices_shape[1], expert_outputs.shape[-1])
        debug_print(self.rank, f"  output_shape={output_shape}")
        
        # CRITICAL VALIDATION before FUSCO call
        expected_send = self.recvindices.numel()  # reversed for combine
        actual_input = expert_outputs.shape[0]
        debug_print(self.rank, f"  VALIDATION: input_tokens={actual_input}, sendindices(recv).numel()={expected_send}")
        
        outputs_unpermuted = _FuscoAllToAll.apply(
            self.fusco, 
            expert_outputs,
            output_shape,
            self.sendindices_unique, 
            self.recvindices, 
            self.send_splits, 
            self.recv_splits)

        debug_print(self.rank, f"  outputs_unpermuted.shape={outputs_unpermuted.shape}")
        return outputs_unpermuted

    def combine_postprocess(self, combined_outputs):
        debug_print(self.rank, f"=== COMBINE POSTPROCESS ===")
        debug_print(self.rank, f"  combined_outputs.shape={combined_outputs.shape}")
        
        combined_outputs = combined_outputs.view(
            self.indices_shape[0], self.indices_shape[1], -1
        )
        debug_print(self.rank, f"  reshaped to: {combined_outputs.shape}")
        
        output = (combined_outputs * self.probs.to(combined_outputs.dtype).unsqueeze(-1)).sum(dim=1)
        debug_print(self.rank, f"  after weighted sum: {output.shape}")
        
        result = output.view(self.hidden_shape)
        debug_print(self.rank, f"  final output: {result.shape}")
        return result


class _FuscoAllToAll(torch.autograd.Function):
    """"Autograd wrapper for FUSCO's all_to_all operation."""
    @staticmethod
    def forward(
        ctx,
        fusco: FUSCO,
        input: torch.Tensor,
        output_shape: tuple,
        recvindices: torch.Tensor,
        sendindices: torch.Tensor,
        recv_splits: torch.Tensor,
        send_splits: torch.Tensor,
    ):
        rank = dist.get_rank()
        debug_print(rank, f"  _FuscoAllToAll.forward:")
        debug_print(rank, f"    input.shape={input.shape}, output_shape={output_shape}")
        debug_print(rank, f"    recvindices.shape={recvindices.shape}, sendindices.shape={sendindices.shape}")
        debug_print(rank, f"    recv_splits={recv_splits.tolist()}, send_splits={send_splits.tolist()}")
        
        # Validate indices vs tensor sizes
        if sendindices.numel() > 0:
            max_send = sendindices.max().item()
            if max_send >= input.shape[0]:
                print(f"[RANK {rank}] FATAL: sendindices max ({max_send}) >= input size ({input.shape[0]})", flush=True)
        
        if recvindices.numel() > 0:
            max_recv = recvindices.max().item()
            if max_recv >= output_shape[0]:
                print(f"[RANK {rank}] FATAL: recvindices max ({max_recv}) >= output size ({output_shape[0]})", flush=True)
        
        assert input.dim() == 2, f"fusco.all_to_all requires 2D input, got {input.dim()}D"
        assert len(output_shape) == 2, f"fusco.all_to_all requires 2D output shape, got {len(output_shape)}"
        
        ctx.fusco = fusco
        ctx.input_shape = input.shape
        ctx.recvindices = recvindices
        ctx.sendindices = sendindices
        ctx.recv_splits = recv_splits
        ctx.send_splits = send_splits
        
        output = input.new_empty(output_shape, dtype=input.dtype, device=input.device)
        
        # Get the stream we'll use for the operation
        current_stream = torch.cuda.current_stream()
        
        debug_print(rank, f"    Calling fusco.all_to_all...")
        
        # CRITICAL: Synchronize before the NCCL call to ensure input data is ready
        current_stream.synchronize()
        
        fusco.all_to_all(
            output=output,
            input=input,
            recvindices=recvindices,
            sendindices=sendindices,
            recv_splits=recv_splits,
            send_splits=send_splits,
            stream=current_stream,
        )
        
        # CRITICAL: Synchronize after NCCL call - NCCL operations are async!
        # Without this, we may read output before NCCL has finished writing to it
        current_stream.synchronize()
        
        debug_print(rank, f"    fusco.all_to_all completed successfully")
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        rank = dist.get_rank()
        fusco = ctx.fusco
        input_shape = ctx.input_shape
        
        debug_print(rank, f"  _FuscoAllToAll.backward:")
        debug_print(rank, f"    grad_output.shape={grad_output.shape}, input_shape={input_shape}")
        
        grad_input = grad_output.new_empty(
            input_shape, dtype=grad_output.dtype, device=grad_output.device
        )
        
        current_stream = torch.cuda.current_stream()
        
        debug_print(rank, f"    Calling fusco.all_to_all (backward)...")
        
        # CRITICAL: Synchronize before NCCL call
        current_stream.synchronize()
        
        fusco.all_to_all(
            output=grad_input,
            input=grad_output,
            recvindices=ctx.sendindices,
            sendindices=ctx.recvindices,
            recv_splits=ctx.send_splits,
            send_splits=ctx.recv_splits,  
            stream=current_stream,
        )
        
        # CRITICAL: Synchronize after NCCL call
        current_stream.synchronize()
        
        debug_print(rank, f"    fusco.all_to_all (backward) completed successfully")
        
        return None, grad_input, None, None, None, None, None
