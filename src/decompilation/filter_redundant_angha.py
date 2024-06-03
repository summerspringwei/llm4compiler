from typing import List
from datasets import Dataset, load_from_disk, concatenate_datasets
from tqdm import tqdm 


def extract_substring(s: str) -> str:
    start = s.find("@") + 1  # find returns the starting index of the substring, add 1 to exclude "@"
    end = s.find("(")
    if start > 0 and end > start:  # both "@" and "(" were found and in correct order
        return s[start:end]
    return None  # return None if "@" or "(" were not found or in wrong order


def get_llvm_ir_func_name(ir: List[str])->str:
    for line in ir:
        line = line.strip()
        if line.startswith("define"):
            return extract_substring(line)
    return None


def main(dataset: Dataset)-> Dataset:
    # 1. First, get the records with same length
    length_records_dict = {}
    for record in tqdm(dataset):
        length = record['input_length']
        if length not in length_records_dict.keys():
            length_records_dict[length] = [record, ]
        else:
            length_records_dict[length].append(record)
    
    # 2. With the records with same length, get the records with same function name
    filtered_length_records_dict = {}
    for length, records in tqdm(length_records_dict.items()):
        func_names_set = set()
        for r in records:
            func_name = get_llvm_ir_func_name(r['output'].split("\n"))
            if func_name not in func_names_set:
                func_names_set.add(func_name)
                if length not in filtered_length_records_dict.keys():
                    filtered_length_records_dict[length] = [r, ]
                else:
                    filtered_length_records_dict[length].append(r)
    
    # 3. Convert the filtered records to the datasets
    filtered_dataset = []
    for length, records in filtered_length_records_dict.items():
        filtered_dataset.extend(records)
    filtered_dataset = Dataset.from_list(filtered_dataset) 
    filtered_dataset = filtered_dataset.sort('input_length')

    return filtered_dataset


def test():
    dataset_train = load_from_disk("/data/xiachunwei/Datasets/decompilation-dataset/AnghaBench-llvm-ir-llc-assembly-O2-seq_length-16K_bbcount-2-average-2_chat_train_sort")
    dataset_train = dataset_train.select(range(0, 200))
    for r in dataset_train:
        print(r['input_length'])
        # print(r['output'])
        print(get_llvm_ir_func_name(r['output'].split("\n")))
    print("=======================================")

    filtered_dataset = main(dataset_train)
    print(len(filtered_dataset))
    for r in filtered_dataset:
        print(r['input_length'])
        # print(r['output'])
        print(get_llvm_ir_func_name(r['output'].split("\n")))
        # print("=======================================")


def run():
    dataset_train = load_from_disk("/data/xiachunwei/Datasets/decompilation-dataset/AnghaBench-llvm-ir-llc-assembly-O2-seq_length-16K_bbcount-2-average-2_chat_train_sort")
    dataset_val = load_from_disk("/data/xiachunwei/Datasets/decompilation-dataset/AnghaBench-llvm-ir-llc-assembly-O2-seq_length-16K_bbcount-2-average-2_chat_val_sort")
    dataset = concatenate_datasets([dataset_train, dataset_val])
    filtered_dataset = main(dataset)
    filtered_dataset.save_to_disk("/data/xiachunwei/Datasets/decompilation-dataset/AnghaBench-llvm-ir-llc-assembly-O2-seq_length-16K_bbcount-2-average-2_chat_all_sort_filtered")
    print(len(dataset), len(filtered_dataset))
    

if __name__ == "__main__":
    # test()
    run()
