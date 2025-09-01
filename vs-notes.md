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
4. [TODO] model parallelism
5. [TODO] llama 8b - fp8

### Next (llama3? TODO)
https://github.com/NVIDIA/Megatron-LM/tree/main/examples/llama




