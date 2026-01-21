export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7   # pick your GPUs

torchrun --standalone \
  --nnodes=1 \
  --nproc_per_node=8 \
  ../../pretrain_gpt.py \
  --distributed-timeout-minutes 60 \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --expert-model-parallel-size 8 \
  --context-parallel-size 1 \
  --expert-tensor-parallel-size 1 \
  --use-distributed-optimizer \
  --overlap-grad-reduce \
  --overlap-param-gather \
  --no-create-attention-mask-in-dataloader \
  --use-mcore-models \
  --sequence-parallel \
  --use-flash-attn \
  --disable-bias-linear \
  --micro-batch-size 2 \
  --global-batch-size 128 \
  --train-iters 100 \
  --exit-duration-in-mins 230 \
  --manual-gc \
  --manual-gc-interval 5 \
  --cross-entropy-loss-fusion \
  --cross-entropy-fusion-impl te \
  --enable-experimental \
  --transformer-impl transformer_engine \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model Qwen/Qwen3-30B-A3B \
  --mock-data \
  --no-mmap-bin-files \
  --num-workers 6 \
  --untie-embeddings-and-output-weights \
  --position-embedding-type rope \
  --rotary-percent 1.0 \
  --rotary-base 1000000 \
  --rotary-seq-len-interpolation-factor 1 \
  --normalization RMSNorm \
  --swiglu \
  --norm-epsilon 1e-06 \
  --num-layers 48 \
  --hidden-size 2048 \
  --ffn-hidden-size 6144 \
  --num-attention-heads 32 \
  --group-query-attention \
  --num-query-groups 4 \
  --kv-channels 128 \
  --qk-layernorm \
  --seq-length 4096 \
  --max-position-embeddings 40960 \
  --make-vocab-size-divisible-by 1187 \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --clip-grad 1.0 \
  --weight-decay 0.1 \
  --lr-decay-iters 20 \
  --lr-warmup-iters 10 \
  --lr 1.2e-4 \
  --min-lr 1.2e-5 \
  --lr-decay-style cosine \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --num-experts 128 \
  --moe-ffn-hidden-size 768 \
  --moe-router-load-balancing-type aux_loss \
  --moe-router-topk 8 \
  --moe-grouped-gemm \
  --moe-aux-loss-coeff 1e-3 \
  --moe-permute-fusion \
  --moe-router-dtype fp32 \
  --moe-router-fusion \
  --eval-iters 32 \
  --eval-interval 500 \
  --auto-detect-ckpt-format \
  --save /root/work/run/todelete/checkpoints \
  --save-interval 500 \
  --dist-ckpt-strictness log_all \
  --init-method-std 0.02 \
  --log-timers-to-tensorboard \
  --log-memory-to-tensorboard \
  --log-num-zeros-in-grad \
  --log-params-norm \
  --log-validation-ppl-to-tensorboard \
  --log-throughput \
  --log-interval 1 \
  --tensorboard-dir /root/work/run/todelete/tensorboard \
  --bf16 \
  --moe-router-force-load-balancing \
  --recompute-granularity selective \
  --recompute-modules moe_act layernorm \
  --moe-token-dispatcher-type alltoall \
  # --moe-token-dispatcher-type flex \
  # --moe-flex-dispatcher-backend deepep

  # --lr-decay-samples 255126953 \
  # --lr-warmup-samples 162761 \
  # --train-samples 268554688 \
  # --data-cache-path /root/work/dev/mcoremoe/mlm-moe-zoo/data_cache \

  # --wandb-project mcoremoe \
  # --wandb-exp-name Qwen3-30B-A3B-TP1PP1EP8CP1VPP1-MBS4GBS256-baseline \

  # --tp-comm-overlap
  # File "/root/work/dev/mcoremoe/megatron/megatron/training/arguments.py", line 935, in validate_args
    # assert args.sequence_parallel == True, 'Tensor parallel communication/GEMM overlap can happen only when sequence parallelism is enabled'

  # --finetune \
  # --load \