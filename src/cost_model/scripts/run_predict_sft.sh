script_directory="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

peft_model_path=/home/xiachunwei/Dataset/HW-cost-model-dataset/cbench_sft_output/sft_lora_model
pretrained_model=/home/xiachunwei/Dataset/CodeLlama-7b-hf
# pretrained_model=/data/xiachunwei/Dataset/codellama/CodeLlama-7b-hf
chinese_tokenizer_path=${script_directory}/../llama_data/cbench_wo_line_no/merged_tokenizer_hf/
output_dir="cbench_sft_predict_output"

torchrun --nnodes 1 --nproc_per_node 1 run_sft_predict.py \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name_or_path ${chinese_tokenizer_path} \
    --output_dir ${output_dir} \
    --torch_dtype float16 \
    --peft_path ${peft_model_path} 
