import sys, os
import logging
import torch
import json

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
    else:
        logger.error("No model is specified!")
    
    logger.info(f"len(tokenizer):{len(tokenizer)}")
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) != embedding_size:
        logger.info("resize the embedding size by the size of the tokenizer")
        model.resize_token_embeddings(len(tokenizer))
    
    if training_args.peft_path is not None:
        logger.info("Peft from pre-trained model")
        model = PeftModel.from_pretrained(model, training_args.peft_path)
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
