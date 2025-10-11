# Make Sure PYTHONPATH point to megatron-lm

install-dependencies:
	pip install datasets==3.6.0
	pip install wandb

prepare-ds-openwebtext-10k:
	rm -rf ./owt-ds
	python hf_ds_to_json.py
	wget -P ./owt-ds https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
	wget -P ./owt-ds https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt

	python ../../tools/preprocess_data.py \
		--input ./owt-ds/openwebtext-10k.jsonl \
		--output-prefix ./owt-ds/openwebtext-10k \
		--vocab-file ./owt-ds/gpt2-vocab.json \
		--tokenizer-type GPT2BPETokenizer \
		--merge-file ./owt-ds/gpt2-merges.txt \
		--workers $(shell nproc) \
		--append-eod

100-gpt2-xl-1gpu-bs1:
	bash ./100_gpt2xl_1gpu_bf16.sh

110-gpt2-xl-ddp-4gpus-gbs4:
	bash ./110_gpt2xl_ddp_4gpus_gbs4.sh

train-dp2-gpt2-large-2gpu:
	bash ./101_train_gpt2_large_dp2.sh

train-tp2-gpt2-large-2gpu:
	bash ./102_train_gpt2_large_tp2.sh

train-tp2-sq-gpt2-large-2gpu:
	bash ./102.1_train_gpt2_large_tp2+sq.sh

train-cp2-gpt2-large-2gpu:
	bash ./103_train_gpt2_large_cp2.sh

train-pp2-gpt2-large-2gpu:
	bash ./104_train_gpt2_large_pp2.sh