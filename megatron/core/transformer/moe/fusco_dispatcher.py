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
        self.num_experts = config.num_moe_experts
        self.topk = config.moe_router_topk
        assert self.tp_size * self.ep_size > 1, "Flex token dispatcher requires TPxEP > 1"

        global GLOBAL_FUSCO
        if GLOBAL_FUSCO is None:
            GLOBAL_FUSCO = FUSCO(
                group_ranks=[dist.get_process_group_ranks(pg_collection.ep)],
                library_path="/workspace/fusco-ep/lib/libfusco.so",
            )
        self.fusco = GLOBAL_FUSCO
        
    def dispatch_preprocess(self, tokens, routing_map, probs):
        # hidden_shape: original hidden states shape [S, B, H]
        # routing_map and probs are output of moe router shape of T, E
        self.hidden_shape = tokens.shape  # Save original shape for restoring later
        
        k_probs, k_indices = torch.topk(probs, self.topk, dim=-1, largest=True, sorted=False)
        k_indices = k_indices.to(torch.int64)

        self.probs = k_probs
        self.indices_shape = k_indices.shape

        # adapting FuscoMoEDispatcher preprocess
        num_local_tokens_per_expert = torch.bincount(k_indices.view(-1), minlength=self.num_experts)

        num_local_tokens_per_rank = num_local_tokens_per_expert.view(
            self.ep_size, self.num_local_experts
        ).sum(dim=1)

        topk = k_indices.size(1)
        flatten_indices = k_indices.view(-1)

        self.sendindices_unique = torch.argsort(flatten_indices, stable=True).contiguous()
        self.sendindices_with_duplicates = (self.sendindices_unique // topk).contiguous()
        self.send_splits = num_local_tokens_per_rank.to(torch.device("cpu"))

        num_global_tokens_per_expert = gather_along_first_dim(
            num_local_tokens_per_expert, self.ep_group
        ).reshape(self.ep_size, self.num_experts)

        num_global_tokens_per_local_expert = num_global_tokens_per_expert[
            :, self.local_expert_indices[0] : self.local_expert_indices[-1] + 1
        ].contiguous()

        num_tokens_per_local_expert = num_global_tokens_per_local_expert.sum(dim=0)

        num_tokens_per_ep = num_global_tokens_per_local_expert.sum(dim=1)

        self.num_ep_tokens = num_global_tokens_per_local_expert.sum()

        if self.num_ep_tokens > 0:
            self.recvindices = idxtools.indices_gen(
                num_global_tokens_per_local_expert,
                num_tokens_per_local_expert,
                num_tokens_per_ep,
                self.num_ep_tokens.item(),
            )
        else:
            self.recvindices = torch.empty(0, dtype=torch.int64, device=torch.cuda.current_device())
        self.recv_splits = num_tokens_per_ep.to(torch.device("cpu"))

        # return num_tokens_per_local_expert
        self.tokens_per_expert = num_tokens_per_local_expert

        return tokens.view(-1, tokens.shape[-1]), k_probs
    
    def token_dispatch(self, hidden_states, _unused_probs):
        # unlike other dispatchers, probs are not dispatched
        # probs will be applied in combine postprocess
        
        # Expand hidden_states to (num_tokens * topk, H) by repeating each token topk times
        # This allows us to use sendindices_unique (unique indices) instead of 
        # sendindices_with_duplicates (duplicate indices), which is necessary for
        # correct gradient accumulation in the backward pass since FUSCO's scatter
        # overwrites rather than accumulates
        num_tokens = hidden_states.shape[0]
        hidden_states_expanded = hidden_states.unsqueeze(1).expand(
            num_tokens, self.topk, -1
        ).reshape(num_tokens * self.topk, -1)
        
        tokens_by_expert = _FuscoAllToAll.apply(
            self.fusco, 
            hidden_states_expanded,
            (self.num_ep_tokens, hidden_states.shape[-1]), # output shape
            self.recvindices, 
            self.sendindices_unique,  # Use unique indices for correct backward
            self.recv_splits, 
            self.send_splits)
        return tokens_by_expert, _unused_probs # probs just dummy to comply to MoELayer
    
    def dispatch_postprocess(self, dispatched_tokens, _unused_probs):
        # dispatched_tokens has been permuted internally in fusco
        # dispatched_tokens.shape[0] == self.tokens_per_expert.sum()?
        # len(self.tokens_per_expert) == self.num_local_experts?
        dummy_probs = torch.ones(dispatched_tokens.shape[0],
                                 dtype=dispatched_tokens.dtype,
                                 device=dispatched_tokens.device)
        return dispatched_tokens, self.tokens_per_expert, dummy_probs
    
    def combine_preprocess(self, expert_outputs):
        # unpermute is done internally in fusco
        # so nothing to preprocess, just forwarding expert_outputs
        return expert_outputs

    def token_combine(self, expert_outputs):
        # unpermute is done internally in fusco
        # Keep output 2D: (num_tokens * topk, H)
        outputs_unpermuted = _FuscoAllToAll.apply(
            self.fusco, 
            expert_outputs,
            (self.indices_shape[0] * self.indices_shape[1], expert_outputs.shape[-1]),  # 2D output
            self.sendindices_unique, 
            self.recvindices, 
            self.send_splits, 
            self.recv_splits)

        return outputs_unpermuted

    def combine_postprocess(self, combined_outputs):
        # combined_outputs is 2D: (num_tokens * topk, H)
        # Reshape to (num_tokens, topk, H) for weighted sum
        combined_outputs = combined_outputs.view(
            self.indices_shape[0], self.indices_shape[1], -1
        )
        # Keep probs in fp32 for precision during multiplication and sum
        # TODO: temporary measure, we can output.to(final_dtype) on return
        # no sure why, so we cast probs to input dtype here
        output = (combined_outputs * self.probs.to(combined_outputs.dtype).unsqueeze(-1)).sum(dim=1)
        
        # Restore original [S, B, H] shape and cast back to input dtype
        return output.view(self.hidden_shape)

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
        assert input.dim() == 2, f"fusco.all_to_all requires 2D input, got {input.dim()}D"
        assert len(output_shape) == 2, f"fusco.all_to_all requires 2D output shape, got {len(output_shape)}"
        # fusco doesn't require 2D output shape, but it will break backward if not 2D, hence the assert
        
        # Save for backward - need to swap indices/splits for the reverse operation
        ctx.fusco = fusco
        ctx.input_shape = input.shape
        ctx.recvindices = recvindices
        ctx.sendindices = sendindices
        ctx.recv_splits = recv_splits
        ctx.send_splits = send_splits
        
        # Create output tensor
        output = input.new_empty(output_shape, dtype=input.dtype, device=input.device)
        
        # Perform FUSCO all_to_all
        fusco.all_to_all(
            output=output,
            input=input,
            recvindices=recvindices,
            sendindices=sendindices,
            recv_splits=recv_splits,
            send_splits=send_splits,
            stream=torch.cuda.current_stream(),
        )
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        fusco = ctx.fusco
        input_shape = ctx.input_shape
        
        # Both input and output are always 2D for FUSCO
        grad_input = grad_output.new_empty(
            input_shape, dtype=grad_output.dtype, device=grad_output.device
        )
        
        # Backward is the reverse of recv-send indices and splits
        fusco.all_to_all(
            output=grad_input,
            input=grad_output,
            recvindices=ctx.sendindices,
            sendindices=ctx.recvindices,
            recv_splits=ctx.send_splits,
            send_splits=ctx.recv_splits,  
            stream=torch.cuda.current_stream(),
        )
        
        return None, grad_input, None, None, None, None, None
