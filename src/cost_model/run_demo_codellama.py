# The following demo code is copied from https://huggingface.co/docs/transformers/main/model_doc/code_llama
import os
from typing import List
from transformers import LlamaForCausalLM, CodeLlamaTokenizer
import torch


def test_codellama():
    # pretrained_model = "/data/xiachunwei/Software/codellama/CodeLlama-7b-hf"
    pretrained_model = "/home/xiachunwei/Dataset/CodeLlama-7b-hf"
    tokenizer = CodeLlamaTokenizer.from_pretrained(pretrained_model)
    model = LlamaForCausalLM.from_pretrained(pretrained_model)
    for k in model.state_dict().keys():
        print(k, model.state_dict()[k].shape, model.state_dict()[k].dtype)
    PROMPT = '''def remove_non_ascii(s: str) -> str:
        """ <FILL_ME>
        return result
    '''
    input_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"]
    generated_ids = model.generate(input_ids, max_new_tokens=128)
    filling = tokenizer.batch_decode(generated_ids[:, input_ids.shape[1]:], skip_special_tokens = True)[0]
    print(PROMPT.replace("<FILL_ME>", filling))


def run_tokenizer(tokenizer: CodeLlamaTokenizer, prompt: str)-> List[int]:
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    return input_ids


def get_tokenizer(file_path: str):
    # tokenizer = CodeLlamaTokenizer.from_pretrained(folder_path)
    tokenizer = CodeLlamaTokenizer(vocab_file=file_path)
    return tokenizer


def get_codellama_tokenizer_and_model(pretrained_model: str):
    tokenizer = CodeLlamaTokenizer.from_pretrained(pretrained_model)
    model = LlamaForCausalLM.from_pretrained(pretrained_model)
    model.eval()
    model.to(torch.device('cuda:0'))
    return tokenizer, model


def test_token_length(file_path: str, tokenizer_folder_path: str):
    tokenizer = get_tokenizer(tokenizer_folder_path)
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if len(line) > 0:
                input_ids = run_tokenizer(tokenizer, line)
                token_list = [tokenizer._convert_id_to_token(int(x)) for x in input_ids[0]]
                # [print(int(x)) for x in input_ids[0]]
                # print(input_ids)
                # print(token_list)
                # print(f"input length: {len(input_ids[0])}")


def get_all_benchmark_token_length(benchmark_dir: str, tokenizer_folder_path: str):
    items = os.listdir(benchmark_dir)
    # Filter out only the directories from the list
    directories = [item for item in items if os.path.isdir(
        os.path.join(benchmark_dir, item))]
    tokenizer = get_tokenizer(tokenizer_folder_path)
    for item in directories:
        file_path = os.path.join(benchmark_dir, item, "random/llm_training_record.json")
        with open(file_path, 'r') as f:
            lines = f.readlines()
            if len(lines) > 0:
                num_elements = torch.numel(tokenizer(lines[0].strip(), return_tensors="pt")["input_ids"][0])
                print(f"{item}: {num_elements // 1024}K") 


if __name__ == "__main__":
    # training_record_file_path = "cBench/network_dijkstra/random/llm_training_record.json"
    # training_record_file_path = "cBench/automotive_susan_c/random/llm_training_record.json"
    # training_record_file_path = "cBench/consumer_tiffmedian/random/llm_training_record.json"
    # tokenizer_path = "merged_tokenizer_sp/cbench_ir_llama.model"
    # test_token_length(training_record_file_path, tokenizer_path)
    test_codellama()
    # get_all_benchmark_token_length("cBench", "merged_tokenizer_sp/cbench_ir_llama.model")
