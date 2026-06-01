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
            num_local_experts (int): The number of local experts.
            router_topk (int): The number of experts for each token to select.
            num_experts (int): The total number of experts in the group.
            config (TransformerConfig): The configuration for the transformer model.
        """
        self.group = group
        self.num_rank_group = dist.get_world_size(self.group)

        self.num_local_experts = num_local_experts
        self.config = config

        self.router_topk = router_topk
        self.num_experts = num_experts
        self.router_dtype = config.moe_router_dtype
        self.capacity_factor = config.moe_expert_capacity_factor
        self.permute_fusion = config.moe_permute_fusion

        # Metadata
        self.token_indices: Optional[torch.Tensor] = None
        self.token_probs: Optional[torch.Tensor] = None
        # Handle used for combine operation
        self.handle = None

        if fused_dispatch is None:
            raise ImportError(
                "DeepEP is not installed. Please install DeepEP package from "
                "https://github.com/deepseek-ai/deepep."
            )
        set_deepep_num_sms(config.moe_deepep_num_sms)

        global _symm_mem_backend_initialized
        if not _symm_mem_backend_initialized:
            symm_mem.set_backend("NVSHMEM")
            _symm_mem_backend_initialized = True
        
        _torch_ver = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
        if _torch_ver <= (2, 11):
            symm_mem.enable_symm_mem_for_group(dist.group.WORLD.group_name)

        self.token_dispatcher = None
        self.probs_dispatcher = None
        self.token_combiner = None      


    def setup_metadata(self, routing_map: torch.Tensor, probs: torch.Tensor):
        num_tokens = routing_map.shape[0]

        routing_map = routing_map.reshape(num_tokens, self.num_experts)
        probs = probs.reshape(num_tokens, self.num_experts)

        self.routing_map = routing_map
        self.probs = probs

        # Convert the format of routing map from multihot to indices.
        self.token_probs, self.token_indices = torch.topk(probs, self.router_topk, dim=-1)
        # Mask the indices of dropped tokens with -1
        if self.capacity_factor is not None:
            mask = self.token_probs == 0
            self.token_indices = self.token_indices.masked_fill(mask, -1)

        if self.token_dispatcher is None:
            # only construct/initialize during the first run as
            # mbs only accessible during runtime 
            # (not during model construction where this manager is instantiated)
            H = self.config.hidden_size
            L = self.config.seq_length
            E = self.num_experts
            K = self.config.moe_router_topk
            EPR = self.num_local_experts
            NRANK_EP = self.num_rank_group
            T = self.routing_map.shape[0]
            ilen = T*K
            olen = int(ilen * 1.25) # int(T*K * EPR * NRANK_EP// E)  # 

            self.token_dispatcher = SymmMemDispatcher(
                                        group = self.group,
                                        align = 1,
                                        in_len = ilen,
                                        out_len = olen,
                                        token_shape = (H,),
                                        num_ranks = NRANK_EP,
                                        num_local_experts = EPR,
                                        dtype = torch.bfloat16,   # TODO: hardcoded for now, need to find a way to access hidden_states
                                        device = torch.cuda.current_device(),
                                    )
            self.prob_dispatcher = SymmMemDispatcher(
                                        group = self.group,
                                        align = 1,
                                        in_len = ilen,
                                        out_len = olen,
                                        token_shape = (),
                                        num_ranks = NRANK_EP,
                                        num_local_experts = EPR,
                                        dtype = probs.dtype,
                                        device = torch.cuda.current_device(),
                                    )
            self.token_combiner = SymmMemCombiner(
                                        group = self.group,
                                        align = 1,
                                        in_len = olen,
                                        out_len = ilen,
                                        token_shape = (H,),
                                        num_ranks = NRANK_EP,
                                        num_local_experts = EPR,
                                        dtype = torch.bfloat16,   # TODO: hardcoded
                                        device = torch.cuda.current_device(),
                                    )

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        probs: torch.Tensor
        # async_finish: bool = False,
        # allocate_on_comm_stream: bool = False,
    ) -> torch.Tensor:
        # DeepEP only supports float32 probs
        if self.token_probs.dtype != torch.float32:
            if self.token_probs.dtype in [torch.bfloat16, torch.float16]:
                logger.warning(
                    "DeepEP only supports float32 probs, please set --moe-router-dtype=fp32"
                )
            self.token_probs = self.token_probs.float()  # downcast or upcast
        # hidden_states, dispatched_indices, dispatched_probs, num_tokens_per_expert, handle = (
        #     fused_dispatch(
        #         hidden_states,
        #         self.token_indices,
        #         self.token_probs,
        #         self.num_experts,
        #         self.group,
        #         async_finish=async_finish,
        #         allocate_on_comm_stream=allocate_on_comm_stream,
        #     )
        # )
        # self.handle = handle
        # self.tokens_per_expert = num_tokens_per_expert
        # self.dispatched_indices = dispatched_indices
        # self.dispatched_probs = dispatched_probs
        insplits = self.routing_map.sum(dim=0)
        dispatched_input = self.token_dispatcher(hidden_states, insplits)
        dispatched_probs = self.prob_dispatcher(probs, insplits)
        return dispatched_input, dispatched_probs

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
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> torch.Tensor:
        # hidden_states, _ = fused_combine(
        #     hidden_states,
        #     self.group,
        #     self.handle,
        #     async_finish=async_finish,
        #     allocate_on_comm_stream=allocate_on_comm_stream,
        # )
        # # Release the handle after combine operation
        # self.handle = None
        hidden_states = self.token_combiner(hidden_states, self.token_dispatcher._out_splits_offsets)
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
            self._comm_manager = _SymmMemManager(
                group=self.tp_ep_group,
                num_local_experts=self.num_local_experts,
                router_topk=self.tp_size * self.config.moe_router_topk,
                num_experts=self.tp_size * self.config.num_moe_experts,
                config=self.config,
            )
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

        self._comm_manager.setup_metadata(routing_map, probs)
        # return hidden_states, self._comm_manager.token_probs
        assert self.config.moe_pad_expert_input_to_capacity is False, "expert_capacity unsupported yet for MoESymmMemTokenDispather"
        (
            permutated_local_input_tokens,
            permuted_probs,
            self.reversed_local_input_permutation_mapping,
            _,
            _,
        ) = permute(
            hidden_states,                         # non K-expanded
            self._comm_manager.routing_map,
            probs=self._comm_manager.probs,
            num_out_tokens=self._comm_manager.routing_map.shape[0] * self.config.moe_router_topk,   # k-expanded #tokens
            fused=self.config.moe_permute_fusion,
            drop_and_pad=False,                    # False for now, else self.config.moe_pad_expert_input_to_capacity
        )
        return permutated_local_input_tokens, permuted_probs

    def token_dispatch(
        self,
        hidden_states: torch.Tensor,
        probs: Optional[torch.Tensor] = None,
        async_finish: bool = True,
        allocate_on_comm_stream: bool = True,
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
        return self._comm_manager.dispatch(hidden_states, probs)

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
        torch.cuda.current_stream().synchronize()
        tokens_per_expert = self._comm_manager.get_number_of_tokens_per_expert()
        ntok = tokens_per_expert.sum().item()
        global_input_tokens = hidden_states[:ntok] 
        permuted_probs = probs[:ntok]
        return global_input_tokens, tokens_per_expert, permuted_probs

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
        async_finish: bool = True,
        allocate_on_comm_stream: bool = True,
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
        return self._comm_manager.combine(hidden_states, async_finish, allocate_on_comm_stream)

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
        torch.distributed.barrier()
        # output = unpermute(
        #     permutated_local_input_tokens,
        #     self.reversed_local_input_permutation_mapping,
        #     restore_shape=self.hidden_shape_before_permute,
        #     routing_map=self.routing_map,
        #     fused=self.config.moe_permute_fusion,
        #     drop_and_pad=self.drop_and_pad,
        # )

        # # Reshape the output tensor
        # output = output.view(self.hidden_shape)
        return hidden_states.view(self.hidden_shape)
