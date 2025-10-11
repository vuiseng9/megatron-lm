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
    * If you use vscode, you may find `megatron-lm/examples/gpt3/vscode/*.json` handy.
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

### Tinkering Training Parallelism with GPT2-XL (1.5B)

|label     |#param|#txblk|#embeding|#heads|
|-         |-     |-     |-        |-     |
|gpt2-large|774M  |36    |1280     |20    |
|gpt2-xl   |1.5B  |48    |1600     |25    |
Both uses 1024 context length. 4x MLP expansion.

We choose 1.5B to align to ZeRO paper discussion.

[to Rewrite] Following experiments are intended setup of 2xGPU with > 80GB, uses nvlink, we use 2xH200 and above. We are gonna start by ablating the parallelism dimension before combining them. 

1. Megatron's design: 
    * `Global batch size (gbs) = micro batch size (mbs) * gradient accumulation steps * data parallel size`. Gradient accumulation steps is automatically calculated if we are required to set `--micro-batch-size` and `global-batch-size`, as well as `data parallel size`. Data parallel size in simple term is number of model replicas. As we start by 1xGPU, data parallel size is 1, keeping accumulation steps to 1. It entails global batch size = micro-batch-size. 
    * How to configure parallelism?
        * `WORLD_SIZE = NNODES * GPUS_PER_NODE =  DP * TP * PP * EP`
        * Replica is automatically derived from WORLD_SIZE / (TP * PP * EP)
        * User set `NNODES, GPUS_PER_NODE`, `TP`, `PP`, `EP` to let mlm derive `DP`.

1. **Warm-up**: `make gpt2-xl-1gpu-bs1` 1x GPU, with no model parallelism, gbs=1, mbs=1. We size it such that it is around 80GB.
    * Key takeaways from logs below:
        1. #parameters per intention: ~1.5B
        1. Theoretical memory footprints: weight and optimizer=26699.47 MB; this is close to ZeRO paper's discussion ("at least 24GB"). TODO: verify that weights and gradients are BF16, optimizer's weight copy in FP32, momentum, FP32, variance, FP32. 16B per parameter.
        1. With batch size of 1, it is consuming >=33GB of GPU memory, which overflows V100 to begin with.
        1. Question: why there is grad norm? just compute but not used?
    ```bash
    Number of parameters in transformer block in billions:  1.47
    Number of parameters in embedding layers in billions: 0.08
    Total number of parameters in billions: 1.56
    Number of parameters in most loaded shard in billions: 1.5554
    compute_activation_memory_without_sp
    Activation memory footprint per transformer layer (precise, without SP): 53.1 MB
    Theoretical memory footprints: weight and optimizer=26699.47 MB, activation=2785.59 MB, total=29485.06 MB
    ```
    ```
    h100x4                    Sat Oct 11 18:11:04 2025  580.82.07
    [0] NVIDIA H100 80GB HBM3 | 38°C,  53 % | 33321 / 81559 MB | root(33314M)
    [1] NVIDIA H100 80GB HBM3 | 27°C,   0 % |     3 / 81559 MB |
    [2] NVIDIA H100 80GB HBM3 | 30°C,   0 % |     3 / 81559 MB |
    [3] NVIDIA H100 80GB HBM3 | 26°C,   0 % |     3 / 81559 MB |
    ```
    * take note of elapse time per iteration (gradient update step)
    ```
    [2025-10-11 18:11:22] iteration      311/  500000 | consumed samples:          311 | elapsed time per iteration (ms): 179.8 | learning rate: 4.339535E-05 | global batch size:     1 | lm loss: 7.220541E+00 | loss scale: 1.0 | grad norm: 6.003 | number of skipped iterations:   0 | number of nan iterations:   0 |
    [2025-10-11 18:11:22] iteration      312/  500000 | consumed samples:          312 | elapsed time per iteration (ms): 180.1 | learning rate: 4.353488E-05 | global batch size:     1 | lm loss: 7.169382E+00 | loss scale: 1.0 | grad norm: 9.685 | number of skipped iterations:   0 | number of nan iterations:   0 |
    [2025-10-11 18:11:22] iteration      313/  500000 | consumed samples:          313 | elapsed time per iteration (ms): 179.9 | learning rate: 4.367442E-05 | global batch size:     1 | lm loss: 7.093079E+00 | loss scale: 1.0 | grad norm: 2.775 | number of skipped iterations:   0 | number of nan iterations:   0 |
    [2025-10-11 18:11:22] iteration      314/  500000 | consumed samples:          314 | elapsed time per iteration (ms): 180.9 | learning rate: 4.381395E-05 | global batch size:     1 | lm loss: 7.408157E+00 | loss scale: 1.0 | grad norm: 2.889 | number of skipped iterations:   0 | number of nan iterations:   0 |
    ```
1. **Vanilla Data Parallel**: `make 110-gpt2-xl-ddp-4gpus-gbs4`. 
    * Key takeaways from logs below:
        1. Memory usage per card is about the same as before because one replica per gpu, mbs=1, gbs=4.
        1. elapsed time per iteration is ~250ms, as opposed to ~180ms before, because of gradient synchronization during optimizer update step.
        1. Exercise for you: make the single card to gbs=4, what is the elapse time per iteration and memory usage? (it should be around 189.1ms, just slight higher because no gradient synchronization overhead, no communication over nvlink/internet. Memory usage should be around 43GB, because of larger batch size)
        1. TODO: find out where and how gradient synchronization is implemented in megatron-lm.
    ```
    h100x4                    Sat Oct 11 18:37:10 2025  580.82.07
    [0] NVIDIA H100 80GB HBM3 | 46°C,  30 % | 33803 / 81559 MB | root(33796M)
    [1] NVIDIA H100 80GB HBM3 | 41°C,  74 % | 33811 / 81559 MB | root(33804M)
    [2] NVIDIA H100 80GB HBM3 | 47°C,  18 % | 33811 / 81559 MB | root(33804M)
    [3] NVIDIA H100 80GB HBM3 | 40°C,  25 % | 33811 / 81559 MB | root(33804M)
    ```
    Constant elapse time per iteration as baseline.
    ```
    [2025-10-11 18:38:41] iteration     1384/  500000 | consumed samples:         5536 | elapsed time per iteration (ms): 251.6 | learning rate: 5.999934E-05 | global batch size:     4 | lm loss: 6.302687E+00 | loss scale: 1.0 | grad norm: 1.591 | number of skipped iterations:   0 | number of nan iterations:   0 |
    [2025-10-11 18:38:41] iteration     1385/  500000 | consumed samples:         5540 | elapsed time per iteration (ms): 250.4 | learning rate: 5.999934E-05 | global batch size:     4 | lm loss: 6.263545E+00 | loss scale: 1.0 | grad norm: 1.350 | number of skipped iterations:   0 | number of nan iterations:   0 |
    [2025-10-11 18:38:42] iteration     1386/  500000 | consumed samples:         5544 | elapsed time per iteration (ms): 250.3 | learning rate: 5.999934E-05 | global batch size:     4 | lm loss: 6.166680E+00 | loss scale: 1.0 | grad norm: 1.392 | number of skipped iterations:   0 | number of nan iterations:   0 |
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