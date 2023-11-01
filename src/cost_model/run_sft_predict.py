import sys, os
import logging
import torch
import json
import copy
from typing import List


from transformers import (
    HfArgumentParser,
    set_seed,
    CodeLlamaTokenizer,
    AutoTokenizer,
    LlamaForCausalLM
)
from typing import List
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.trainer_utils import get_last_checkpoint
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    PeftModel,
    get_peft_model_state_dict
)

from sft_args import (
    ModelArguments,
    MyTrainingArguments,
)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


logger = logging.getLogger(__name__)

# base_model.model.model.embed_tokens.weight torch.Size([32510, 4096]) torch.float16
# base_model.model.lm_head.weight torch.Size([32510, 4096]) torch.float16
# Still need these four weight
# "base_model.model.model.embed_tokens.original_module.weight", 
# "base_model.model.model.embed_tokens.modules_to_save.default.weight", 
# "base_model.model.lm_head.original_module.weight", 
# "base_model.model.lm_head.modules_to_save.default.weight".

def merge_lora_and_codellama_stat_dict(lora_model_state_dict, codellama_model_state_dict, lora_prfix="default", modules_to_save: List[str]=None):
    new_state_dict = {}
    def add_default_prefix(name: str):
        com = name.split(".")
        com.insert(-1, lora_prfix)
        return ".".join(com)
    
    for k, v in lora_model_state_dict.items():
        new_state_dict.update({add_default_prefix(k): v})
        if modules_to_save is not None:
            for module_name in modules_to_save:
                if k.find(module_name) != -1:
                    original_com = k.split(".")
                    com = copy.deepcopy(original_com)
                    com.insert(-1, "original_module")
                    new_k = ".".join(com)
                    new_state_dict.update({new_k: v})
                    com = copy.deepcopy(original_com)
                    com.insert(-1, "modules_to_save.default")
                    new_k = ".".join(com)
                    new_state_dict.update({new_k: v})
        
    for k, v in codellama_model_state_dict.items():
        k = "base_model.model." + k
        if k not in new_state_dict.keys():
            new_state_dict.update({k: v})
    
    return new_state_dict



# Setup logging
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,  # if training_args.local_rank in [-1, 0] else logging.WARN,
    handlers=[logging.StreamHandler(sys.stdout)],)

def get_tokenizer_and_model():
    parser = HfArgumentParser((ModelArguments, MyTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, training_args = parser.parse_args_into_dataclasses()
    
    # # Detecting last checkpoint.
    # if os.path.isdir(training_args.output_dir):
    #     last_checkpoint = get_last_checkpoint(training_args.output_dir)
    #     if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
    #         raise ValueError(
    #             f"Output directory ({training_args.output_dir}) already exists and is not empty. "
    #             "Use --overwrite_output_dir to overcome."
    #         )
    
    if model_args.tokenizer_name_or_path:
        tokenizer = CodeLlamaTokenizer.from_pretrained(model_args.tokenizer_name_or_path)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    
    if tokenizer.pad_token is None:
        print(f"Adding pad token {DEFAULT_PAD_TOKEN}")
        tokenizer.add_special_tokens(dict(pad_token=DEFAULT_PAD_TOKEN))
    base_model_state_dict = {}
    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        model = LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True
        )
        base_model_state_dict = model.state_dict()
    else:
        logger.error("No model is specified!")
    
    logger.info(f"original len(tokenizer):{len(tokenizer)}")
    embedding_size = model.get_input_embeddings().weight.shape[0]
    logger.info(f"CodeLlama embedding_size:{embedding_size}")
    if len(tokenizer) != embedding_size:
        logger.info("resize the embedding size by the size of the tokenizer")
        model.resize_token_embeddings(len(tokenizer))
    
    if training_args.peft_path is not None:
        # logger.info("Peft from pre-trained model")
        # model = PeftModel.from_pretrained(model, training_args.peft_path)
        logger.info("Init new peft model")
        target_modules = training_args.trainable.split(',')
        modules_to_save = training_args.modules_to_save
        if modules_to_save is not None:
            modules_to_save = modules_to_save.split(',')
        lora_rank = training_args.lora_rank
        lora_dropout = training_args.lora_dropout
        lora_alpha = training_args.lora_alpha
        logger.info(f"target_modules: {target_modules}")
        logger.info(f"lora_rank: {lora_rank}")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
            inference_mode=False,
            r=lora_rank, lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            modules_to_save=modules_to_save)
        model = get_peft_model(model, peft_config)
        # Load from pre-trained state dict
        peft_state_dict = torch.load(training_args.peft_path)
        for k in peft_state_dict.keys():
            logger.info(f"peft_state_dict: {k}, {peft_state_dict[k].shape}, {peft_state_dict[k].dtype}")

        # We manully merge the state dict
        mergered_state_dict = merge_lora_and_codellama_stat_dict(peft_state_dict, base_model_state_dict, modules_to_save=modules_to_save)
        model.load_state_dict(mergered_state_dict, strict=False)
        # base_model.model.model.embed_tokens.weight torch.Size([32510, 4096]) torch.float16
        # base_model.model.lm_head.weight torch.Size([32510, 4096]) torch.float16
        # Still need these four weight
        # "base_model.model.model.embed_tokens.original_module.weight", 
        # "base_model.model.model.embed_tokens.modules_to_save.default.weight", 
        # "base_model.model.lm_head.original_module.weight", 
        # "base_model.model.lm_head.modules_to_save.default.weight".
        # embedding_size = model.get_input_embeddings().weight.shape[0]
        # logger.info(f"Peft embedding_size:{embedding_size}")
    else:
        logger.error("No peft model is specified!")
    model.eval()
    model.to(torch.device('cuda:0'))
    return tokenizer, model


def run_model_inference(tokenizer: PreTrainedTokenizer, model: LlamaForCausalLM, prompt: str, device=torch.device('cuda:0')) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    generated_ids = model.generate(
        input_ids = input_ids,
        max_new_tokens=128,
        attention_mask = attention_mask,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id)
    filling = tokenizer.batch_decode(generated_ids[:, input_ids.shape[1]:], skip_special_tokens = True)[0]

    return filling


def cbench_cost_model_format(record: dict)->str:
    return f"""Below is an instruction that describes a task. "
            Write a output that appropriately completes the request.\n\n"
            instruction: { {record['instruction:']} },\n code1: { {record['code1:']} },\n code2: { {record['code2:']} }\n output: <FILL_ME>"""

def get_assembly_dataset_record(file_path: str) -> List[str]:
    prompt_list, label_list = [], []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            record = json.loads(line)
            prompt_list.append(cbench_cost_model_format(record))
            label_list.append(record['output'])
    return prompt_list, label_list


def main():
    tokenizer, model = get_tokenizer_and_model()
    test_file_path = "/home/xiachunwei/Dataset/HW-cost-model-dataset/cBench/telecom_adpcm_d/random/llm_training_record.json"
    prompt_list, label_list = get_assembly_dataset_record(test_file_path)
    for prompt, label in zip(prompt_list, label_list):
        # print(prompt)
        response = run_model_inference(tokenizer, model, prompt)
        print(f"predict: {response}, label: {label}")


if __name__ == "__main__":
    main()
