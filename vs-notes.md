## Megatron-LM

* r01: 2025/09/01
* r00: 2025/07/13

Per recommendation, it is indeed better to use nvidia docker image because installing transformer-engine is cumbersome, requiring stars to align. Torch, Cuda, Cudnn, TE. Details: Attempted to set up megatron-lm in conda but failed due to specific dependencies with transformer engine and dependencies are hard to installed properly. CUDA is relatively easily, cuDNN is a nightmare, pytorch must be compiled from scratch (do you know the configurations? I don't, also not worth the time to figure out), and finally compiling Transformer Engine. if you just want to user of megatron-lm, don't go this path. if you figure out, pls share with me. Ice-creams, my treat :).

```bash
d-nv-run -v /root/work:/root/work nvcr.io/nvidia/pytorch:25.06-py3
```

### [DO NOT USE] First Tiny Example (it doesn't seem to work now Sept 1, keeping here so we avoid it)
<details>
[Official Quick Start and step-by-step guide to Megatron-LM](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/index.html#quick-start)

```bash
git clone https://github.com/NVIDIA/Megatron-LM
```
`examples/run_simple_mcore_train_loop.py` creates a sample GPT model split across tensors (Tensor model parallel) on 2 GPUS, and run a forward pass through it using a MockGPT dataset helper class that we created in Megatron Core.

Run
```bash
cd Megatron-LM
PYTHONPATH=$PYTHON_PATH:. torchrun --nproc-per-node 2 examples/run_simple_mcore_train_loop.py
```
</details>

### [Best to Getting Started] Train a Tiny GPT3
Based on [ootb examples](https://github.com/vuiseng9/megatron-lm/blob/main/examples/gpt3/README.md), training gpt3 345M and 857M. Verified on `main branch: f4fa7d6b commit` with docker image below, on 2xB200.

1. Setup.
    * `d-nv-run -v /root/work:/root/work nvcr.io/nvidia/pytorch:25.06-py3`
    * `git clone https://github.com/vuiseng9/megatron-lm && cd megatron-lm && git checkout 250901-ootb`, no installation required. Do remember to configure PYTHONPATH to include `<pathto>/megatron-lm`
    * active directory `megatron-lm/examples/gpt3`, we use Makefile to capture steps to reproduce.
    * `make install-dependencies` 
2. Tiny dataset preparation (I forgot where I reference/cross-reference it). 
    * `make prepare-ds-openwebtext-10k`
    * Megatron requires its format of dataset. This step downloads a huggingface dataset, a small one, dumping the dataset into json and finally tokenized dataset is generated, see `openwebtext-10k_text_document.bin` and `openwebtext-10k_text_document.idx` in `owt-ds`.
3. Launch Training, start small, 1x gpu, no model parallelism.
    * 345m gpt3, use wandb logging (do login, change entity (account) to yours). Gotcha: wandb training metric logging to wandb depends on tensorboard (internal implementation), make sure we turn on tensorboard! I spent a great deal figure out the design of logging which imho not good.
    * `wandb login`
    * `make train-gpt3-345m-1gpu-x-model-parallelism`
    * `code --diff train_gpt3_175b_distributed.sh 00_train_gpt3_345m.sh` Review changes to ootb launch script.

### Tinkering Model Parallelism with GPT2-large (775M)

|label     |#param|#txblk|#embeding|#heads|
|-         |-     |-     |-        |-     |
|gpt2-large|774M  |36    |1280     |20    |
|gpt2-xl   |1.5B  |48    |1600     |25    |
Both uses 1024 context length. 4x MLP expansion.

Following experiments are intended setup of 2xGPU with > 80GB, uses nvlink, we use 2xH200 and above. We are gonna start by ablating the parallelism dimension before combining them. 

1. `Global batch size = micro batch size * gradient accumulation steps * data parallel size`. We skip gradient accumulation steps for now. Gradient accumulation steps is automatically calculated if we set micro-batch-size and global-batch-size. That means we need to set global batch size and micro-batch-size, as well as data parallel size. Data parallel size in simple term is number of model replicas. As we start by 1xGPU, data parallel size is 1, keeping accumulation steps to 1. It entails global batch size = micro-batch-size. 

1. **Baseline**: `make train-baseline-gpt2-large-1gpu` 1x GPU, no model parallelism, micro-batch-size=global-batch-size=32 (no accumulate gradient). We size it such that it is around 80GB.
    * expected logs:
    ```bash
    [before the start of training step] datetime: 2025-09-07 23:53:58 
    Number of parameters in transformer block in billions:  0.71
    Number of parameters in embedding layers in billions: 0.06
    Total number of parameters in billions: 0.77
    Number of parameters in most loaded shard in billions: 0.7724
    ```
    ```
    dc-h200x2       Sun Sep  7 23:57:26 2025  580.82.07
    [0] NVIDIA H200 | 60°C, 100 % | 80161 / 143771 MB | root(80154M)
    [1] NVIDIA H200 | 26°C,   0 % |     3 / 143771 MB |
    ```
    * take note of elapse time per iteration (gradient update step)
    ```
    [2025-09-07 23:58:47] iteration      609/  500000 | consumed samples:        19488 | elapsed time per iteration (ms): 438.3 | learning rate: 5.999998E-05 | global batch size:    32 | lm loss: 6.159759E+00 | loss scale: 131072.0 | grad norm: 1.095 | number of skipped iterations:   0 | number of nan iterations:   0 |
    [2025-09-07 23:58:47] iteration      610/  500000 | consumed samples:        19520 | elapsed time per iteration (ms): 436.7 | learning rate: 5.999998E-05 | global batch size:    32 | lm loss: 6.002230E+00 | loss scale: 131072.0 | grad norm: 0.791 | number of skipped iterations:   0 | number of nan iterations:   0 |
    ```
1. **Data Parallel**: `make train-dp2-gpt2-large-2gpu`. How to configure data parallelism?
    * WORLD_SIZE = NNODES * GPUS_PER_NODE =  DP * TP * PP * EP
    * Replica is automatically derived from WORLD_SIZE / (TP * PP * EP)
    * for this experiment, we just set `nproc-per-node=2 and global batch size to 32x2` and no model parallelism, so DP=2, TP=1, PP=1, EP=1. Replica = 2/1 = 2.
    * Expected outcomes:
    ```
    [0] NVIDIA H200 | 59°C,  93 % | 80543 / 143771 MB | root(80536M)
    [1] NVIDIA H200 | 58°C,  88 % | 80543 / 143771 MB | root(80536M)
    ```
    Constant elapse time per iteration as baseline.
    ```
    [2025-09-08 00:19:09] iteration       26/  500000 | consumed samples:         1664 | elapsed time per iteration (ms): 436.9 | learning rate: 1.395349E-06 | global batch size:    64 | lm loss: 1.029183E+01 | loss scale: 131072.0 | grad norm: 7.831 | number of skipped iterations:   0 | number of nan iterations:   0 |
    [2025-09-08 00:19:09] iteration       27/  500000 | consumed samples:         1728 | elapsed time per iteration (ms): 437.5 | learning rate: 1.534884E-06 | global batch size:    64 | lm loss: 1.026151E+01 | loss scale: 131072.0 | grad norm: 6.683 | number of skipped iterations:   0 | number of nan iterations:   0 |
    ```
    This pretrain script has fixed the gradient update steps, means with 2xglobal batch size, it is effectively 2x more epochs.

1. **Tensor Parallel**: `make train-tp2-gpt2-large-2gpu`. How to configure tensor parallelism?
    * We set `--tensor-model-parallel-size 2` and `--micro-batch-size 64` (because we only have one replica due to each shard per gpu, and we want to keep the same effective work per gradient update to dp2 case, i.e. 64 global batch size)
    * Expected outcomes:
    ```
    [0] NVIDIA H200 | 63°C, 100 % | 85969 / 143771 MB | root(85960M)
    [1] NVIDIA H200 | 56°C, 100 % | 85967 / 143771 MB | root(85960M)
    ```
    Communication overhead is expected, elapse time per iteration is expected to be higher than baseline and dp2 case. Communication happens between gpus during forward and backward pass.
    ```
    [2025-09-08 01:07:45] iteration      177/  500000 | consumed samples:        11328 | elapsed time per iteration (ms): 542.8 | learning rate: 2.246512E-05 | global batch size:    64 | lm loss: 7.733203E+00 | loss scale: 131072.0 | grad norm: 2.639 | number of skipped iterations:   0 | number of nan iterations:   0 |
    [2025-09-08 01:07:45] iteration      178/  500000 | consumed samples:        11392 | elapsed time per iteration (ms): 543.6 | learning rate: 2.260465E-05 | global batch size:    64 | lm loss: 7.707379E+00 | loss scale: 131072.0 | grad norm: 2.785 | number of skipped iterations:   0 | number of nan iterations:   0 |
    ```

1. **Sequence Parallel**: `make train-sp2-gpt2-large-2gpu`. How to configure sequence parallelism?
    * We set `--sequence-parallel` and `--tensor-model-parallel-size 2` and `--micro-batch-size 64` (because we only have one replica, we need keep each GPU memory usage similar to baseline) and `--global-batch-size 64` (to keep same effective batch size as Data Parallel). 
    * So DP=1, TP=2, PP=1, EP=1. Replica = 2/2 = 1.
    * Expected outcomes:
    better gpu memory usage than tp2 case, because sequence parallel offloads some of the activation memory to cpu.
    ```
    [0] NVIDIA H200 | 59°C, 100 % | 71571 / 143771 MB | root(71562M)
    [1] NVIDIA H200 | 55°C,  89 % | 71569 / 143771 MB | root(71562M)
    ```
    ```
    [2025-09-08 04:09:11] iteration      147/  500000 | consumed samples:         9408 | elapsed time per iteration (ms): 538.9 | learning rate: 1.827907E-05 | global batch size:    64 | lm loss: 8.350144E+00 | loss scale: 131072.0 | grad norm: 2.711 | number of skipped iterations:   0 | number of nan iterations:   0 |
    [2025-09-08 04:09:11] iteration      148/  500000 | consumed samples:         9472 | elapsed time per iteration (ms): 541.4 | learning rate: 1.841860E-05 | global batch size:    64 | lm loss: 8.275106E+00 | loss scale: 131072.0 | grad norm: 2.853 | number of skipped iterations:   0 | number of nan iterations:   0 |
    ```

1. **Context Parallel**: `make train-cp2-gpt2-large-2gpu`

    * Expected outcomes:
    [0] NVIDIA H200 | 66°C, 100 % | 81917 / 143771 MB | root(81908M)
    [1] NVIDIA H200 | 59°C, 100 % | 81557 / 143771 MB | root(81548M)
    Seem to have faster elapse time per iteration than TP and SP case, why?
    ```
    [2025-09-08 04:35:16] iteration      547/  500000 | consumed samples:        35008 | elapsed time per iteration (ms): 514.5 | learning rate: 5.999999E-05 | global batch size:    64 | lm loss: 6.092395E+00 | loss scale: 131072.0 | grad norm: 1.094 | number of skipped iterations:   0 | number of nan iterations:   0 |
    [2025-09-08 04:35:16] iteration      548/  500000 | consumed samples:        35072 | elapsed time per iteration (ms): 519.1 | learning rate: 5.999999E-05 | global batch size:    64 | lm loss: 6.211164E+00 | loss scale: 131072.0 | grad norm: 1.892 | number of skipped iterations:   0 | number of nan iterations:   0 |
    ```
1. **Pipeline Parallel**: `make train-pp2-gpt2-large-2gpu`. How to configure pipeline parallelism?
    * We set `--pipeline-model-parallel-size 2` and `--micro-batch-size 64` (because we only have one replica, we need keep each GPU memory usage similar to baseline) and `--global-batch-size 64` (to keep same effective batch size as Data Parallel). 
    * So DP=1, TP=2, PP=1, EP=1. Replica = 2/2 = 1.
    * Expected outcomes:
    Questions: where is the split? why memory usage uneven?
    ```
    [0] NVIDIA H200 | 45°C, 100 % | 61249 / 143771 MB | root(61240M)
    [1] NVIDIA H200 | 52°C, 100 % | 88299 / 143771 MB | root(88290M)
    ```
    Elapse time per iteration is expected to be higher than baseline due to communication overhead. (how do we observe this overhead?). In DP case, communication is only during gradient update step, in PP case, communication happens during forward and backward pass.
    ```
    [2025-09-08 00:50:11] iteration      239/  500000 | consumed samples:        15296 | elapsed time per iteration (ms): 821.7 | learning rate: 3.111628E-05 | global batch size:    64 | lm loss: 6.909464E+00 | loss scale: 131072.0 | grad norm: 1.742 | number of skipped iterations:   0 | number of nan iterations:   0 |
    [2025-09-08 00:50:12] iteration      240/  500000 | consumed samples:        15360 | elapsed time per iteration (ms): 813.0 | learning rate: 3.125581E-05 | global batch size:    64 | lm loss: 6.941336E+00 | loss scale: 131072.0 | grad norm: 1.321 | number of skipped iterations:   0 | number of nan iterations:   0 |
    ```
    To come back not very solid.

 [TODO]


how dataset is handled

4. [TODO] model parallelism
5. [TODO] llama 8b - fp8

### Next (llama3? TODO)
https://github.com/NVIDIA/Megatron-LM/tree/main/examples/llama


For training
DP first
TP next
PP last

Inference is different, PP first.


where exactly the sharding happens?
DP: just wrap with DDP? what is in ddp wrapper? handling the weight update?
TP/Squence parallel: wrap on linear and embedding layer? so when we do sequence parallel, also in linear layers?
PP: how? how to break into stages? 


what metric

../llama - pretrain-gpt
../pretrain_gpt.py, mixtral
../t5 pretrain_t5.py
../bert - pretrain_bert.py
post-training