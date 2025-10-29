/root/work/dev/mlm/megatron-lm/megatron/training/arguments.py

def maybe_wrap_for_inprocess_restart(pretrain):

    args = arguments.parse_args(ignore_unknown_args=True)

def add_megatron_arguments(parser: argparse.ArgumentParser):

def _add_distributed_args(parser):


for i, g in enumerate(parser._action_groups):
    #print(i, g.title)
    if g.title == "distributed":
        for a in g._group_actions:
            print(a.option_strings)


### DP
    --overlap-grad-reduce
    --ddp-num-buckets
    --ddp-bucket-size
    --ddp-pad-buckets-for-high-nccl-busbw
    --ddp-average-in-collective
    --overlap-param-gather
    --overlap-param-gather-with-optimizer-step
    --use-distributed-optimizer
    --use-nccl-ub
    --use-megatron-fsdp
    --data-parallel-sharding-strategy {
        no_shard,
        optim = ZeRO1,
        optim_grads = ZeRO2,
        optim_grads_params = ZeRO3}
    --no-gradient-reduce-div-fusion
    --fsdp-double-buffer
    --suggested-communication-unit-size
    --keep-fp8-transpose-cache
    --enable-full-sharding-in-hsdp
    --num-distributed-optimizer-instances
    --use-torch-fsdp2
    --torch-fsdp2-no-reshard-after-forward

### TP/SP
--tensor-model-parallel-size
[deprecated]--model-parallel-size

### CP
--context-parallel-size 
--cp-comm-type (p2p, a2a, allgather or a2a+p2p)
--hierarchical-context-parallel-sizes 

### PP/VPP
    --pipeline-model-parallel-size
    --decoder-first-pipeline-num-layers
    --decoder-last-pipeline-num-layers
    --pipeline-model-parallel-layout
    --num-layers-per-virtual-pipeline-stage
    --num-virtual-stages-per-pipeline-rank
    --microbatch-group-size-per-virtual-pipeline-stage

    --no-overlap-p2p-communication
    --overlap-p2p-communication-warmup-flush
    --account-for-embedding-in-pipeline-split

### PP+DP
    --no-align-grad-reduce (async PP grad-reduce right before optimizer step)
    --no-align-param-gather (async PP param-gather right before forward)
    --defer-embedding-wgrad-compute
    --wgrad-deferral-limit

### PP+TP
    --no-scatter-gather-tensors-in-pipeline

### Dist Setup
--distributed-backend
--distributed-timeout-minutes
--local-rank
--lazy-mpu-init
--nccl-communicator-config-path
--use-tp-pp-dp-mapping : from tp-cp-ep-dp-pp to tp-cp-ep-pp-dp.
for fault tolerance
--replication
--replication-jump
--replication-factor
--use-ring-exchange-p2p
--account-for-loss-in-pipeline-split
comms 
--use-sharp
--sharp-enabled-group
--init-model-with-meta-device









