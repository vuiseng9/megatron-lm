import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import torch

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
from megatron.core.transformer.moe.token_dispatcher import MoETokenDispatcher, _DispatchManager

import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from megatron.core.transformer.moe.symm_mem_dispatch import TokenDispatcher as SymmMemDispatcher
from megatron.core.transformer.moe.symm_mem_combine import TokenCombiner as SymmMemCombiner

logger = logging.getLogger(__name__)

_symm_mem_backend_initialized = False


class SymmMem2DA2A(torch.autograd.Function):
    """Differentiable all_to_all_vdev_2d (the MoE *dispatch*).

    forward : inp (rank-major, dense) -> out (expert-major, major_align-padded)
    backward: grad_out (expert-major, padded) -> grad_inp (rank-major, dense),
              realised with all_to_all_vdev_2d_offset (combine).

    This is the mirror of SymmMem2DA2AOffset: dispatch and combine are adjoints of
    each other, so dispatch's backward is exactly combine routed on the gradient.
    The combine in backward is fed the *same* out_splits_offsets that dispatch
    produced in forward, which is precisely the padded layout grad_out lives in.

    Returns three tensors:
        out                : [total_padded, H] expert-major, padded activations (differentiable)
        out_splits_offsets : [2, E] int64 (row0 splits, row1 offsets)        (non-diff)
        offs               : [n_local_experts] int32 grouped_mm group ends    (non-diff)
    """

    @staticmethod
    def forward(ctx, inp, in_splits, group_name, major_align,
                n_local_experts, max_in_numel, max_out_numel):
        # inp        : [m, H] rank-major dense (packed by destination global expert)
        # in_splits  : [E] int64 rank-major splits (index g = dst_rank*EPR + local_e)
        device, H = inp.device, inp.shape[1]
        E = in_splits.shape[0]
        ws = E // n_local_experts

        inp_symm = symm_mem.empty(max_in_numel, H, dtype=inp.dtype, device=device).zero_()
        inp_symm[: inp.shape[0]].copy_(inp)
        in_splits_symm = symm_mem.empty(E, dtype=torch.int64, device=device).copy_(in_splits)
        out_symm = symm_mem.empty(max_out_numel, H, dtype=inp.dtype, device=device).zero_()
        out_so = symm_mem.empty((2, E), dtype=torch.int64, device=device).fill_(-1)

        torch.cuda.synchronize(device)
        dist.barrier()

        torch.ops.symm_mem.all_to_all_vdev_2d(
            inp_symm, out_symm, in_splits_symm, out_so, group_name, major_align=major_align
        )

        # Per-expert padded sizes -> grouped_mm group offsets; total_padded == used rows.
        recv = out_so[0]                                  # [E] expert-major true splits
        per_pad = torch.clamp(
            (recv.view(n_local_experts, ws).sum(dim=1) + major_align - 1)
            // major_align * major_align,
            min=major_align,
        )
        offs = per_pad.cumsum(0).to(torch.int32)          # [n_local_experts]
        total_padded = int(offs[-1].item())

        ctx.group_name = group_name
        ctx.max_in_numel = max_in_numel
        ctx.max_out_numel = max_out_numel
        ctx.in_shape = tuple(inp.shape)
        ctx.save_for_backward(out_so.detach().clone())    # the padded layout grad_out lives in

        return out_symm[:total_padded].clone(), out_so.detach().clone(), offs.detach().clone()

    @staticmethod
    def backward(ctx, grad_out, _grad_so, _grad_offs):
        # grad_out : [total_padded, H] grad wrt the (expert-major, padded) output.
        (out_so,) = ctx.saved_tensors
        device, H = grad_out.device, grad_out.shape[1]
        E = out_so.shape[1]

        # backward == COMBINE(grad_out): expert-major padded -> rank-major dense.
        g_symm = symm_mem.empty(ctx.max_out_numel, H, dtype=grad_out.dtype, device=device).zero_()
        g_symm[: grad_out.shape[0]].copy_(grad_out)
        in_so = symm_mem.empty((2, E), dtype=torch.int64, device=device).copy_(out_so)
        grad_inp_symm = symm_mem.empty(ctx.max_in_numel, H, dtype=grad_out.dtype, device=device).zero_()
        out_so_bwd = symm_mem.empty((2, E), dtype=torch.int64, device=device).fill_(-1)

        torch.cuda.synchronize(device)
        dist.barrier()

        torch.ops.symm_mem.all_to_all_vdev_2d_offset(
            g_symm, grad_inp_symm, in_so, out_so_bwd, ctx.group_name
        )

        m = ctx.in_shape[0]
        grad_inp = grad_inp_symm[:m].clone()
        # grads for: inp, in_splits, group_name, major_align, n_local_experts, max_in_numel, max_out_numel
        return grad_inp, None, None, None, None, None, None


class SymmMem2DA2AOffset(torch.autograd.Function):
    """Differentiable all_to_all_vdev_2d_offset (the MoE *combine*).

    forward : inp (expert-major, major_align-padded) -> out (rank-major, dense)
    backward: grad_out (rank-major, dense)            -> grad_inp (expert-major,
              major_align-padded), realised with all_to_all_vdev_2d (dispatch).

    Autograd-visible tensors (inp / grad_inp / out) are ordinary tensors and may
    have rank-varying lengths; only the *internal* NVSHMEM comm buffers must be a
    constant size across ranks, hence the explicit `max_in_numel` / `max_out_numel`.
    """

    @staticmethod
    def forward(ctx, inp, in_splits_offsets, group_name, major_align,
                max_in_numel, max_out_numel):
        # inp                : [m, H]  data to combine (expert-major, padded). m may vary per rank.
        # in_splits_offsets  : [2, E]  int64 -- row0 input splits, row1 input offsets (the padded layout)
        device, H = inp.device, inp.shape[1]
        E = in_splits_offsets.shape[1]

        # Stage the differentiable inputs into constant-size symmetric buffers.
        inp_symm = symm_mem.empty(max_in_numel, H, dtype=inp.dtype, device=device).zero_()
        inp_symm[: inp.shape[0]].copy_(inp)
        in_so = symm_mem.empty((2, E), dtype=torch.int64, device=device)
        in_so.copy_(in_splits_offsets)
        out_symm = symm_mem.empty(max_out_numel, H, dtype=inp.dtype, device=device).zero_()
        out_so = symm_mem.empty((2, E), dtype=torch.int64, device=device).fill_(-1)

        torch.cuda.synchronize(device)
        dist.barrier()

        torch.ops.symm_mem.all_to_all_vdev_2d_offset(
            inp_symm, out_symm, in_so, out_so, group_name
        )
        out_numel = int(out_so[0].sum().item())

        # The rank-major splits the combine produced == the splits dispatch consumes
        # as input on the way back. That + major_align reproduces `in_splits_offsets`.
        ctx.group_name = group_name
        ctx.major_align = major_align
        ctx.max_in_numel = max_in_numel
        ctx.max_out_numel = max_out_numel
        ctx.in_shape = tuple(inp.shape)
        ctx.save_for_backward(out_so[0].detach().clone())

        return out_symm[:out_numel].clone(), out_so.detach().clone()

    @staticmethod
    def backward(ctx, grad_out, _grad_out_so):
        # grad_out : [out_numel, H] grad wrt the (rank-major, dense) combine output.
        (rank_major_splits,) = ctx.saved_tensors
        device, H = grad_out.device, grad_out.shape[1]
        E = rank_major_splits.shape[0]

        # backward == DISPATCH(grad_out): rank-major dense -> expert-major padded.
        g_symm = symm_mem.empty(ctx.max_out_numel, H, dtype=grad_out.dtype, device=device).zero_()
        g_symm[: grad_out.shape[0]].copy_(grad_out)
        in_splits = symm_mem.empty(E, dtype=torch.int64, device=device).copy_(rank_major_splits)
        grad_inp_symm = symm_mem.empty(ctx.max_in_numel, H, dtype=grad_out.dtype, device=device).zero_()
        bwd_so = symm_mem.empty((2, E), dtype=torch.int64, device=device).fill_(-1)

        torch.cuda.synchronize(device)
        dist.barrier()

        torch.ops.symm_mem.all_to_all_vdev_2d(
            g_symm, grad_inp_symm, in_splits, bwd_so, ctx.group_name,
            major_align=ctx.major_align,
        )

        m = ctx.in_shape[0]
        grad_inp = grad_inp_symm[:m].clone()  # padding rows stay 0 -> 0 grad, as required
        # grads for: inp, in_splits_offsets, group_name, major_align, max_in_numel, max_out_numel
        return grad_inp, None, None, None, None, None


class _SymmMemManager(_DispatchManager):
    """
    A manager class to handle fused all-to-all communication processes for MoE models using
    DeepEP backend. See https://github.com/deepseek-ai/deepep for more details.

    The workflow of the DeepEP dispatcher is:
    (1) setup_metadata(): Process routing map and probabilities to prepare dispatch metadata
    (2) dispatch():
        - Use fused kernel to permute tokens and perform all-to-all communication in single step
    (3) get_permuted_hidden_states_by_instances():
        - Convert routing map and probabilities to multihot format
        - Permute tokens using fused kernel
    (4) get_restored_hidden_states_by_instances():
        - Reverse permutation using fused kernel
    (5) combine():
        - Reverse process using fused kernel to unpermute and perform all-to-all in single step

    This implementation uses fused communication kernels (fused_dispatch/fused_combine) that
    combine permutation and communication operations for improved efficiency compared to
    separate permute+alltoall steps.
    """

    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        max_tokens_per_rank: int,
        major_align: int,
        num_local_experts: int,
        router_topk: int,
        num_experts: int,
        config: TransformerConfig,
    ):
        """
        Initialize the DeepEP dispatcher.

        Args:
            group (torch.distributed.ProcessGroup): The process group to use for communication.
                This should be the ETPxEP group.
            max_tokens_per_rank (int): The maximum number of tokens per rank.
            major_align (int): The alignment for major dimensions.
            num_local_experts (int): The number of local experts.
            router_topk (int): The number of experts for each token to select.
            num_experts (int): The total number of experts in the group.
            config (TransformerConfig): The configuration for the transformer model.
        """
        self.group = group
        self.group_name = group.group_name
        self.num_rank_group = dist.get_world_size(self.group)

        self.num_local_experts = num_local_experts
        self.config = config

        self.major_align = major_align
        self.max_in = max_tokens_per_rank
        self.max_out = (
            2 * max_tokens_per_rank + (num_local_experts + 1) * major_align
        )

        self.router_topk = router_topk
        self.num_experts = num_experts
        self.router_dtype = config.moe_router_dtype
        self.capacity_factor = config.moe_expert_capacity_factor
        self.permute_fusion = config.moe_permute_fusion

        global _symm_mem_backend_initialized
        if not _symm_mem_backend_initialized:
            symm_mem.set_backend("NVSHMEM")
            _symm_mem_backend_initialized = True
        
        _torch_ver = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
        if _torch_ver <= (2, 11):
            symm_mem.enable_symm_mem_for_group(dist.group.WORLD.group_name)



    def setup_metadata(self, routing_map: torch.Tensor, probs: torch.Tensor):
        pass

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        probs: torch.Tensor,
        in_splits: torch.Tensor,

    ) -> torch.Tensor:

        dispatched_tokens, self.dispatched_tokens_so, _  = SymmMem2DA2A.apply(
            hidden_states, in_splits, self.group_name, self.major_align,
            self.num_local_experts, self.max_in, self.max_out,
        )
        
        dispatched_probs, dispatched_probs_so, _  = SymmMem2DA2A.apply(
            probs.unsqueeze(-1), in_splits, self.group_name, self.major_align,
            self.num_local_experts, self.max_in, self.max_out,
        )
        return dispatched_tokens, dispatched_probs

    def _indices_to_multihot(self, indices, probs):
        """
        Converts a tensor of indices to a multihot vector.

        Args:
            indices (torch.Tensor): [num_tokens, topk] token indices, where -1 means masked out.
            probs (torch.Tensor): [num_tokens, topk] token probabilities.

        Returns:
            A tuple of (routing_map, probs), where routing_map is the multihot vector
            and probs is the multihot probabilities.
        """
        batch_size = indices.shape[0]
        multihot_routing_map = torch.zeros(
            (batch_size, self.num_local_experts), dtype=torch.long, device=indices.device
        )

        multihot_probs = torch.zeros(
            (batch_size, self.num_local_experts), dtype=torch.float, device=indices.device
        )

        mask = indices != -1
        valid_indices = indices[mask]
        row_indices = torch.arange(batch_size, device=indices.device).repeat_interleave(
            mask.sum(dim=1)
        )
        multihot_routing_map[row_indices, valid_indices] = 1
        multihot_probs[row_indices, valid_indices] = probs[mask]
        return multihot_routing_map.bool(), multihot_probs

    def get_number_of_tokens_per_expert(self) -> torch.Tensor:
        """
        Get the number of tokens per expert.
        """
        # l0_from_rank0, l0_from_rank1, ... l1_from_rank0, l1_from_rank1, 
        return self.token_dispatcher._out_splits_offsets[0].view(self.num_local_experts, self.num_rank_group).sum(dim=1)
         

    def combine(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:

        hidden_states, _ = SymmMem2DA2AOffset.apply(
            hidden_states, self.dispatched_tokens_so, self.group_name, self.major_align,
            self.max_out, self.max_in,
        )
        return hidden_states

    def _pad_routing_map(
        self, routing_map: torch.Tensor, tokens_per_expert: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pad the routing map to the nearest multiple of the pad_multiple.
        """
        pad_multiple = get_align_size_for_quantization(self.config)

        num_input_tokens = routing_map.shape[0]
        target_tokens_per_expert = (
            torch.ceil(tokens_per_expert / pad_multiple) * pad_multiple
        ).long()

        # Check if there are enough tokens to pad
        enough_tokens_to_pad = torch.all(target_tokens_per_expert <= num_input_tokens)
        if not enough_tokens_to_pad:
            logger.warning(
                "Not enough tokens to pad. The total number of tokens received in this rank "
                "is smaller than the target number of tokens for each expert. "
                "Falling back to explicit padding within GroupedMLP"
            )
        else:
            if is_experimental_enabled() and self.permute_fusion:
                from megatron.core.fusions.fused_pad_routing_map import fused_pad_routing_map

                routing_map = fused_pad_routing_map(routing_map, pad_multiple)
            else:
                routing_map = pad_routing_map(routing_map, pad_multiple)
            tokens_per_expert = target_tokens_per_expert
        return routing_map, tokens_per_expert

    def get_permuted_hidden_states_by_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if is_experimental_enabled() and self.permute_fusion:
            self.dispatched_routing_map, self.dispatched_probs = fused_indices_to_multihot(
                self.dispatched_indices, self.dispatched_probs, self.num_local_experts
            )
        else:
            self.dispatched_routing_map, self.dispatched_probs = self._indices_to_multihot(
                self.dispatched_indices, self.dispatched_probs
            )
        if self.config.moe_router_padding_for_quantization:
            self.dispatched_routing_map, self.tokens_per_expert = self._pad_routing_map(
                self.dispatched_routing_map, self.tokens_per_expert
            )

        self.hidden_shape_before_permute = hidden_states.shape
        assert self.dispatched_probs.dtype == torch.float32, "DeepEP only supports float32 probs"
        (
            hidden_states,
            permuted_probs,
            self.reversed_mapping_for_combine,
            self.pad_offsets,
            self.tokens_per_expert,
        ) = permute(
            hidden_states,
            self.dispatched_routing_map,
            probs=self.dispatched_probs,
            num_out_tokens=self.tokens_per_expert.sum().item(),
            fused=self.permute_fusion,
            tokens_per_expert=self.tokens_per_expert,
            align_size=get_align_size_for_quantization(self.config),
        )
        if self.router_dtype == "fp64":
            permuted_probs = permuted_probs.to(torch.float64)
        return hidden_states, permuted_probs

    def get_restored_hidden_states_by_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = unpermute(
            hidden_states,
            self.reversed_mapping_for_combine,
            restore_shape=self.hidden_shape_before_permute,
            routing_map=self.dispatched_routing_map,
            fused=self.permute_fusion,
            pad_offsets=self.pad_offsets,
        )
        return hidden_states


class MoESymmMemTokenDispatcher(MoETokenDispatcher):
    """TODO A flexible token dispatcher that abstracts the underlying tensor and expert
    parallelism. It uses a single communication group over all TP and EP ranks,
    making the dispatch logic independent of the specific parallelism strategy.
    """

    def __init__(
        self,
        num_local_experts: int,
        local_expert_indices: List[int],
        config: TransformerConfig,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        """
        Initialize the Flex token dispatcher.

        Args:
            num_local_experts (int): Number of local experts on the current device.
            local_expert_indices (List[int]): Indices of local experts on the current device.
            config (TransformerConfig): Configuration for the transformer model.
            pg_collection (ProcessGroupCollection, optional): Process groups for MoE operations.
        """
        super().__init__(config=config, pg_collection=pg_collection)

        self.num_local_experts = num_local_experts
        self.local_expert_indices = local_expert_indices
        assert self.tp_size * self.ep_size > 1, "Flex token dispatcher requires TPxEP > 1"
        if self.config.moe_flex_dispatcher_backend == "torch":
            self._comm_manager = None # constructed on first forward since token length not available at init
            self.cudagraph_attrs = ['_comm_manager.token_probs', '_comm_manager.token_indices']
        else:
            raise ValueError(
                f"Invalid backend: {self.config.moe_flex_dispatcher_backend}"
                "Please set --moe-flex-dispatcher-backend=symm_mem or "
                "--moe-flex-dispatcher-backend=torch"
            )

    def set_shared_experts(self, shared_experts):
        raise NotImplementedError(
            "Shared expert overlap is not supported in Flex Token Dispatcher."
        )

    def _initialize_metadata(self, routing_map: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        """
        Initialize the routing map and probs to a unified format covering the TPxEP group.
        This design decouples the communication group from underlying model parallelism groups,
        such that the communication strategy of tokens can be agnostic of TP size and EP size.

        This function expands the routing_map from shape [num_local_tokens, num_experts] to
        [num_local_tokens, world_size, num_local_experts]. Each element in the routing_map
        indicates whether a token should be sent to a specific rank. Specifically, the
        routing_map is replicated across TP group since each TP ranks in a TP group should
        receive the same tokens.
        """
        num_local_tokens = routing_map.shape[0]
        world_size = self.tp_size * self.ep_size
        # Organize routing map and probs to [num_local_tokens, world_size, num_local_experts]
        routing_map = (
            routing_map.reshape(num_local_tokens, self.ep_size, 1, self.num_local_experts)
            .expand(-1, -1, self.tp_size, -1)
            .reshape(num_local_tokens, world_size, self.num_local_experts)
        ).contiguous()
        probs = (
            probs.reshape(num_local_tokens, self.ep_size, 1, self.num_local_experts)
            .expand(-1, -1, self.tp_size, -1)
            .reshape(num_local_tokens, world_size, self.num_local_experts)
        ).contiguous()
        return routing_map, probs

    # @jit_fuser
    def dispatch_preprocess(
        self, hidden_states: torch.Tensor, routing_map: torch.Tensor, probs: torch.Tensor
    ):
        """Initializes routing metadata and prepares tensors for fused dispatch.

        This method reshapes input tensors and processes routing information into a
        unified format, where the routing map is expanded to cover the TPxEP communication domain,
        enabling the token dispatch logic to be agnostic to parallelism strategies.

        Args:
            hidden_states (torch.Tensor): Input hidden states to be processed
            routing_map (torch.Tensor): Map indicating which expert each token should be routed to
            probs (torch.Tensor): Routing probabilities for each token-expert pair

        Returns:
            A tuple of reshaped hidden states and token probabilities.
        """
        self.hidden_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])

        # Initialize metadata
        routing_map, probs = self._initialize_metadata(routing_map, probs)

        # self._comm_manager.setup_metadata(routing_map, probs)
        num_tokens = routing_map.shape[0]
        routing_map = routing_map.reshape(num_tokens, self.config.num_moe_experts)
        probs = probs.reshape(num_tokens, self.config.num_moe_experts)

        # Convert the format of routing map from multihot to indices.
        token_probs, token_indices = torch.topk(probs, self.config.moe_router_topk, dim=-1)
        # Mask the indices of dropped tokens with -1
        if self.config.moe_expert_capacity_factor is not None:
            mask = token_probs == 0
            token_indices = token_indices.masked_fill(mask, -1)

        flat_k_eids = token_indices.reshape(-1)
        # in_splits
        self.ntok_per_eid = torch.bincount(flat_k_eids, minlength=self.config.num_moe_experts)

        k_expanded_tok_ids_by_eid_order = flat_k_eids.argsort()
        # for post combine unpermutation
        self.inv_perm = k_expanded_tok_ids_by_eid_order.argsort()

        tok_ids_by_eid_order = k_expanded_tok_ids_by_eid_order // self.config.moe_router_topk # T*K
        slot_by_eid_order    = k_expanded_tok_ids_by_eid_order % self.config.moe_router_topk # T*K

        permutated_tokens = hidden_states[tok_ids_by_eid_order]
        permutated_probs = token_probs[tok_ids_by_eid_order, slot_by_eid_order]

        if self._comm_manager is None:
        #     # only construct/initialize during the first run as
        #     # mbs only accessible during runtime 
        #     # (not during model construction where this manager is instantiated)
        #     H = self.config.hidden_size
        #     L = self.config.seq_length
        #     E = self.num_experts
        #     K = self.config.moe_router_topk
        #     EPR = self.num_local_experts
        #     NRANK_EP = self.num_rank_group
        #     T = self.routing_map.shape[0]
        #     ilen = T*K
        #     olen = int(ilen * 1.25) # int(T*K * EPR * NRANK_EP// E)  # 
            self._comm_manager = _SymmMemManager(
                group=dist.group.WORLD,
                max_tokens_per_rank=permutated_tokens.shape[0],
                major_align=1,
                num_local_experts=self.num_local_experts,
                router_topk=self.config.moe_router_topk,
                num_experts=self.config.num_moe_experts,
                config=self.config,
            )
        # return hidden_states, self._comm_manager.token_probs
        assert self.config.moe_pad_expert_input_to_capacity is False, "expert_capacity unsupported yet for MoESymmMemTokenDispather"
        return permutated_tokens, permutated_probs

    def token_dispatch(
        self,
        hidden_states: torch.Tensor,
        probs: Optional[torch.Tensor] = None,
    ):
        """
        Execute fused permutation and AlltoAll communication.

        This method currently leverages DeepEP's fused dispatch kernel, which combines token
        permutation and AlltoAll communication into a single optimized operation.
        The fused approach reduces memory bandwidth requirements and enables better
        overlap between computation and communication operations.

        Args:
            hidden_states (torch.Tensor): Preprocessed hidden states to be dispatched
            probs (torch.Tensor): Routing probabilities (unused in current implementation)
            async_finish (bool): Whether to use asynchronous communication completion
            allocate_on_comm_stream (bool): Whether to allocate buffers on communication stream

        Returns:
            A tuple of dispatched tokens and probabilities.
        """
        return self._comm_manager.dispatch(hidden_states, probs, self.ntok_per_eid)

    def dispatch_postprocess(self, hidden_states: torch.Tensor, probs: torch.Tensor):
        """Converts dispatched tokens to a per-expert format for expert processing.

        This method transforms the output of the fused dispatch into the tensor
        organization required for the expert computation.

        Args:
            hidden_states (torch.Tensor): Hidden states after fused dispatch
            probs (torch.Tensor): Routing probabilities after fused dispatch

        Returns:
            A tuple of permuted tokens, token counts per expert, and permuted probabilities.
        """
        # _out_splits_offsets is written as a side-effect of all_to_all_vdev_2d (not returned),
        # so PyTorch has no dependency edge to it. NVSHMEM remote puts from peer ranks land
        # asynchronously; synchronize the stream here to ensure the A2A kernel (and its
        # NVSHMEM quiet/barrier) has fully completed before we read _out_splits_offsets.
        tokens_per_expert = self._comm_manager.dispatched_tokens_so[0].view(
                                self.num_local_experts, self._comm_manager.num_rank_group).sum(dim=1)
        return hidden_states, tokens_per_expert, probs.squeeze()

    def combine_preprocess(self, hidden_states: torch.Tensor):
        """Pre-processes hidden states before combining them after expert processing.

        This method restores the hidden states to their original ordering before expert processing
        by using the communication manager's restoration function.
        """
        # hidden_states = self._comm_manager.get_restored_hidden_states_by_experts(hidden_states)
        return hidden_states

    def token_combine(
        self,
        hidden_states: torch.Tensor,
    ):
        """Executes fused un-permutation and communication using DeepEP kernels.

        This is the inverse of the `token_dispatch` operation.

        Args:
            hidden_states (torch.Tensor): Expert outputs ready for combination
            async_finish (bool): Whether to use asynchronous communication completion
            allocate_on_comm_stream (bool): Whether to allocate buffers on communication stream

        Returns:
            Combined tokens after fused un-permutation and communication.
        """
        return self._comm_manager.combine(hidden_states)

    def combine_postprocess(self, hidden_states: torch.Tensor):
        """
        Restores the original tensor shape and finalizes the MoE layer output.

        This method performs the final step of the MoE token processing pipeline
        by reshaping the combined tokens back to their original input dimensions.

        Args:
            hidden_states (torch.Tensor): Combined tokens.

        Returns:
            The final MoE layer output reshaped to its original dimensions.
        """
        # output = output.view(self.hidden_shape)
        return hidden_states[self.inv_perm].reshape(
            -1, self.config.moe_router_topk, self.hidden_shape[-1]).sum(dim=1).reshape(self.hidden_shape)
    
         