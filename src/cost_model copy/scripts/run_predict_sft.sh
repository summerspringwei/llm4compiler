script_directory="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

lora_rank=8
lora_alpha=32
lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
modules_to_save="embed_tokens,lm_head"
lora_dropout=0.05
# peft_model_path=/home/xiachunwei/Dataset/HW-cost-model-dataset/cbench_sft_output/sft_lora_model
peft_model_path=/home/xiachunwei/Dataset/cbench_sft_output_dbg_2/perf_lora_model_state_dict.pt
pretrained_model=/home/xiachunwei/Dataset/CodeLlama-7b-hf
# pretrained_model=/data/xiachunwei/Dataset/codellama/CodeLlama-7b-hf
chinese_tokenizer_path=${script_directory}/../llama_data/cbench_wo_line_no/merged_tokenizer_hf/
output_dir="cbench_sft_predict_output"

torchrun --nnodes 1 --nproc_per_node 1 run_sft_predict.py \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name_or_path ${chinese_tokenizer_path} \
    --output_dir ${output_dir} \
    --peft_path ${peft_model_path}\
    --lora_rank ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --trainable ${lora_trainable} \
    --modules_to_save ${modules_to_save} \
    --lora_dropout ${lora_dropout} \
    --bf16 true \
    | tee predict.log 2>&1
