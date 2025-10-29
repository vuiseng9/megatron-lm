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

# DP
110-gpt2-xl-ddp-4gpus-gbs4:
	bash ./110_gpt2xl_ddp_4gpus_gbs4.sh

111-gpt2-xl-dp-zero2-4gpus-gbs4:
	bash ./111_gpt2xl_zero2_4gpus_gbs4.sh

# TP Weak Scaling
211-weak-scaling-tp1-gpt2-1.2B:
	bash ./211_weak_tp1_gpt2-1.2B.sh

212-weak-scaling-tp2-gpt2-2.5B:
	bash ./212_weak_tp2_gpt2-2.5B.sh

214-weak-scaling-tp4-gpt2-4.2B:
	bash ./214_weak_tp4_gpt2-4.2B.sh

218-weak-scaling-tp8-gpt2-8.3B:
	bash ./218_weak_tp8_gpt2-8.3B.sh

# Weak Scaling TP + SP
sp.211-weak-scaling-tp1-gpt2-1.2B:
	bash ./211_weak_tp1_gpt2-1.2B.sh 1

sp.212-weak-scaling-tp2-gpt2-2.5B:
	bash ./212_weak_tp2_gpt2-2.5B.sh 1

sp.214-weak-scaling-tp4-gpt2-4.2B:
	bash ./214_weak_tp4_gpt2-4.2B.sh 1

sp.218-weak-scaling-tp8-gpt2-8.3B:
	bash ./218_weak_tp8_gpt2-8.3B.sh 1

# TP Strong Scaling
221-strong-scaling-tp1-gpt2-1.2B:
	bash ./220_strong_tpN_gpt2-1.2B.sh 1

222-strong-scaling-tp2-gpt2-1.2B:
	bash ./220_strong_tpN_gpt2-1.2B.sh 2

224-strong-scaling-tp4-gpt2-1.2B:
	bash ./220_strong_tpN_gpt2-1.2B.sh 4

228-strong-scaling-tp8-gpt2-1.2B:
	bash ./220_strong_tpN_gpt2-1.2B.sh 8

# Strong Scaling TP + SP
sp.221-strong-scaling-tp1-gpt2-1.2B:
	bash ./220_strong_tpN_gpt2-1.2B.sh 1 1

sp.222-strong-scaling-tp2-gpt2-1.2B:
	bash ./220_strong_tpN_gpt2-1.2B.sh 2 1

sp.224-strong-scaling-tp4-gpt2-1.2B:
	bash ./220_strong_tpN_gpt2-1.2B.sh 4 1

sp.228-strong-scaling-tp8-gpt2-1.2B:
	bash ./220_strong_tpN_gpt2-1.2B.sh 8 1

# CP
301-cp1-gpt2xl-gbs4-len4096-oom:
	bash ./300_cpN_gpt2xl_gbs4_len4096.sh 1

302-cp2-gpt2xl-gbs4-len4096:
	bash ./300_cpN_gpt2xl_gbs4_len4096.sh 2

304-cp4-gpt2xl-gbs4-len4096:
	bash ./300_cpN_gpt2xl_gbs4_len4096.sh 4

308-cp8-gpt2xl-gbs4-len4096:
	bash ./300_cpN_gpt2xl_gbs4_len4096.sh 8

311-gpt2xl-gbs4-len8192-cp8:
	bash ./310_gpt2xl_gbsX_lenX_cpX.sh 4 8192 8

312-gpt2xl-gbs4-len16384-cp8:
	bash ./310_gpt2xl_gbsX_lenX_cpX.sh 4 16384 8

313-gpt2xl-gbs8-len8192-cp8:
	bash ./310_gpt2xl_gbsX_lenX_cpX.sh 8 8192 8

314-gpt2xl-gbs8-len8192-cp4-oom:
	bash ./310_gpt2xl_gbsX_lenX_cpX.sh 8 16384 4