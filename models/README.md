# Pre-train
In this example, we show how to do pre-training using retromae with the FlagEmbedding libiary.

## 1. Installation
* **with pip**
```
pip install -U FlagEmbedding
```

* **from source**
```
git clone https://github.com/FlagOpen/FlagEmbedding.git
cd FlagEmbedding
pip install  .
```
For development, install as editable:
```
pip install -e .
```

## 2. Train

```bash
torchrun --nproc_per_node 1 \
-m FlagEmbedding.baai_general_embedding.retromae_pretrain.run \
--output_dir ./pretrain_model \
--model_name_or_path BAAI/bge-small-zh-v1.5 \
--train_data pretrain.jsonl \
--learning_rate 2e-5 \
--num_train_epochs 25 \
--per_device_train_batch_size 32 \
--dataloader_drop_last True \
--max_seq_length 512 \
--logging_steps 10 \
--dataloader_num_workers 12
```

# Finetune
In this example, we show how to do fine-tuning with the FlagEmbedding libiary.

## Train
torchrun --nproc_per_node 1 \
-m FlagEmbedding.baai_general_embedding.finetune.run \
--output_dir ./finetune_model \
--model_name_or_path ./BGE_S_continual_pretrained \
--train_data ./finetune.jsonl \
--learning_rate 3e-5 \
--fp16 \
--num_train_epochs 30 \
--per_device_train_batch_size 32 \
--dataloader_drop_last True \
--normlized True \
--temperature 0.02 \
--query_max_len 512 \
--passage_max_len 256 \
--train_group_size 2 \
--negatives_cross_device \
--logging_steps 10 \
--save_steps 1000 \
--query_instruction_for_retrieval ""