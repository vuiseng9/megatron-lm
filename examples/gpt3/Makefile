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

train-gpt3-345m-1gpu-x-model-parallelism:
	bash ./00_train_gpt3_345m.sh