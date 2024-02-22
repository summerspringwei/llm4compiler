
import os
import json
from tqdm import tqdm
from transformers import AutoTokenizer

# def remove_quation_marks(s: str) -> str:
#     while (s.startswith('"') and s.startswith("'")) or (s.startswith("'") and s.startswith('"')):
#         s = s[1:-1]
#     return s

def load_json(file_path: str, path_key: str, content_key: str):
    datasets = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines, desc='Processing', unit='item'):
            record = json.loads(line)
            while isinstance(record, str):
                record = json.loads(record)
            # Remove the file extension
            base_path = os.path.splitext(record[path_key])[0]
            datasets[base_path] = record[content_key]
    return datasets


def load_tokenizer(model_path = "/workspace/Dataset/TinyLlama-1.1B-step-50K-105b/"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    return tokenizer


def merge_two_json(model_path, assembly_file, c_file, merged_file, total_length_threshold = 0):
    content_key = 'text'
    assembly_dataset = load_json(assembly_file, "assembly_file", content_key)
    c_dataset = load_json(c_file, "c_file", content_key)
    
    # Merge two dataset
    merged_dataset = {}

    tokenizer = load_tokenizer(model_path)
    for key, value in tqdm(assembly_dataset.items(), desc='Processing', unit='item'):
        if key in c_dataset:
            merged_dataset[key] = {
                "assembly": value,
                "assembly_length": len(tokenizer.encode(value)),
                "c": c_dataset[key],
                "c_length": len(tokenizer.encode(c_dataset[key]))
            }
    with open(merged_file, 'w') as f:
        for key, value in tqdm(merged_dataset.items(), desc='Processing', unit='item'):
            if total_length_threshold is not None and \
                total_length_threshold > 0 and \
                value['assembly_length'] + value['c_length'] < total_length_threshold:
                f.write(json.dumps({key: value}) + '\n')


def merge(model_path = "/data0/xiachunwei/Dataset/TinyLlama-1.1B-step-50K-105b/",
    assembly_file = "/data0/xiachunwei/Dataset/decompilation-dataset/AnghaBench-assembly-g-O2.json",
    c_file = "/data0/xiachunwei/Dataset/decompilation-dataset/AnghaBench-C.json",
    merged_file = "/data0/xiachunwei/Dataset/AnghaBench_paired_assembly-g-O2_C.json"):
    # model_path = "/data0/xiachunwei/Dataset/TinyLlama-1.1B-step-50K-105b/"
    # assembly_file = "/data0/xiachunwei/Dataset/decompilation-dataset/head_AnghaBench-assembly-g-O2.json"
    # c_file = "/data0/xiachunwei/Dataset/decompilation-dataset/head_AnghaBench-C.json"
    # merged_file = "/data0/xiachunwei/Dataset/head_AnghaBench_paired_assembly-g-O2_C.json"
    merge_two_json(model_path, assembly_file, c_file, merged_file, 2048)


def prepare_train(merged_file, output_file):
    with open(merged_file, 'r') as f:
        lines = f.readlines()
        with open(output_file, 'w') as f:
            for line in tqdm(lines, desc='Processing', unit='item'):
                record = json.loads(line)
                while isinstance(record, str):
                    record = json.loads(record)
                
                for k, v in record.items():
                    out_record = {"text": f"assembly: {v['assembly']} \n### c: {v['c']}",
                        "file:": k
                    }
                f.write(json.dumps(out_record) + '\n')


def prepare_chat(merged_file, output_file):
    out_records = []
    with open(merged_file, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines, desc='Processing', unit='item'):
                record = json.loads(line)
                while isinstance(record, str):
                    record = json.loads(record)
                for k, v in record.items():
                    out_record = {"instruction": "decompile x86 assembly to C.",
                        "input": v['assembly'],
                        "output": v['c'],
                        "file:": k
                    }
                    out_records.append(out_record)
    json.dump(out_records, open(output_file, "w"), indent=4, sort_keys=True, separators=(',', ':'))

if __name__=="__main__":
    # merge()
    # prepare_train("/data0/xiachunwei/Dataset/decompilation-dataset/AnghaBench_paired_assembly-g-O2_C_2K.json", 
    #     "/data0/xiachunwei/Dataset/decompilation-dataset/AnghaBench_text_train_paired_assembly-g-O2_C_2K.json")

    prepare_chat("/data0/xiachunwei/Dataset/decompilation-dataset/AnghaBench_paired_assembly-g-O2_C_2K.json", 
                 "/data0/xiachunwei/Dataset/decompilation-dataset/AnghaBench_instruct_train_paired_assembly-g-O2_C_2K.json")

