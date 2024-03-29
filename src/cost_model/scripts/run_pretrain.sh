#!/bin/bash

# This script is copied from run_sft.sh

lr=1e-4
lora_rank=8
lora_alpha=32
lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
modules_to_save="embed_tokens,lm_head"
lora_dropout=0.05

script_directory="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

pretrained_model=/home/xiachunwei/Dataset/CodeLlama-7b-hf
# pretrained_model=/data/xiachunwei/Dataset/codellama/CodeLlama-7b-hf
# pretrained_model=/home/xiachunwei/Dataset/CodeLlama-7b-hf
chinese_tokenizer_path=${script_directory}/../llama_data/cbench_wo_line_no/merged_tokenizer_hf/

# dss location
# dataset_dir=${script_directory}/../cBench/telecom_adpcm_d/random/

# 101 location
# dataset_dir=/data/xiachunwei/Dataset/HW-cost-model-dataset/cBench/telecom_adpcm_d/random/
# 42 location
# dataset_dir=/home/xiachunwei/Dataset/HW-cost-model-dataset/cBench/telecom_adpcm_d/random/
dataset_dir=/home/xiachunwei/Dataset/HW-cost-model-dataset/pretrain_cbench_examples

per_device_train_batch_size=4
per_device_eval_batch_size=1
gradient_accumulation_steps=8
# output_dir=/home/xiachunwei/Dataset/cbench_sft_output_on_full_telecom_adpcm_d/
output_dir=/home/xiachunwei/Dataset/cbench_pretrain_on_examples/
# peft_model=
# validation_file=${script_directory}/../cBench/telecom_adpcm_d/random/llm_training_record.json
# validation_file=/home/xiachunwei/Dataset/HW-cost-model-dataset/cBench/telecom_adpcm_d/random/llm_training_record.json
# validation_file=/home/xiachunwei/Dataset/HW-cost-model-dataset/pretrain_cbench_examples/all_benchmark_pre_train.txt
validation_file=${dataset_dir}

deepspeed_config_file=scripts/ds_zero2_no_offload.json

# Set this if debug
# --max_steps 2 \

torchrun --nnodes 1 --nproc_per_node 2 training/run_clm_pretrain_with_peft.py \
    --deepspeed ${deepspeed_config_file} \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name_or_path ${chinese_tokenizer_path} \
    --dataset_dir ${dataset_dir} \
    --validation_split_percentage 0.001 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --do_train \
    --do_eval \
    --seed $RANDOM \
    --num_train_epochs 1 \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.03 \
    --weight_decay 0 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_total_limit 3 \
    --evaluation_strategy steps \
    --eval_steps 100 \
    --save_steps 200 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --preprocessing_num_workers 8 \
    --max_seq_length 10240 \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --lora_rank ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --trainable ${lora_trainable} \
    --modules_to_save ${modules_to_save} \
    --lora_dropout ${lora_dropout} \
    --bf16 true \
    --torch_dtype bfloat16 \
    --validation_file ${validation_file} \
    --gradient_checkpointing \
    --ddp_find_unused_parameters False | tee train.log 2>&1
