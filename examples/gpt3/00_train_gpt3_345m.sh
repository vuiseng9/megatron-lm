#!/bin/bash

# Runs the "175B" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=1
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

RUNDIR="./sandbox_run"
CHECKPOINT_PATH=$RUNDIR/ckpt #<Specify path>
TENSORBOARD_LOGS_PATH=$RUNDIR/tb #<Specify path>
mkdir -p $CHECKPOINT_PATH $TENSORBOARD_LOGS_PATH

MLMROOT="../../"
export PYTHONPATH=$MLMROOT:$PYTHONPATH

DSDIR="./owt-ds"
VOCAB_FILE=$DSDIR/gpt2-vocab.json
MERGE_FILE=$DSDIR/gpt2-merges.txt
DATA_PATH=$DSDIR/openwebtext-10k_text_document

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

GPT_MODEL_ARGS=(
    --num-layers 12 
    --hidden-size 512 
    --num-attention-heads 8 
    --seq-length 1024 
    --max-position-embeddings 1024 
    --attention-backend auto # Can use (flash/fused/unfused/local)
)

    # --rampup-batch-size 16 16 5859375 
TRAINING_ARGS=(
    --micro-batch-size 1 
    --global-batch-size 1536 
    --train-iters 500000 
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    --fp16
    --lr 6.0e-5 
    --lr-decay-style cosine 
    --min-lr 6.0e-6
    --lr-warmup-fraction .001 
    --lr-decay-iters 430000 
)

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size 1 
	--pipeline-model-parallel-size 1 
)

DATA_ARGS=(
    --data-path $DATA_PATH 
    --vocab-file $VOCAB_FILE 
    --merge-file $MERGE_FILE 
    --split 949,50,1
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 5
    --save-interval 10000 
    --eval-interval 1000 
    --save $CHECKPOINT_PATH 
    --load $CHECKPOINT_PATH 
    --eval-iters 10
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
    --wandb-entity vchua
    --wandb-project mlm-sandbox
    --wandb-exp-name $(date +"%y%m%d_%H%M%S")_gpt3_345m_gpu${GPUS_PER_NODE}_tp${MODEL_PARALLEL_ARGS[1]}_pp${MODEL_PARALLEL_ARGS[3]}
)

echo "Executing..."
echo "torchrun ${DISTRIBUTED_ARGS[@]} $MLMROOT/pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}"

torchrun ${DISTRIBUTED_ARGS[@]} $MLMROOT/pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}
