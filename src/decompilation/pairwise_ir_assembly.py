import os
import json
import pickle
import subprocess
from pathlib import Path
from multiprocessing import Pool
import tqdm

import fire
import transformers
from transformers import AutoTokenizer

from common import custom_logging
from common import preprocessing_utils
from common import draw

logger = custom_logging.get_custom_logger()


def compile_llc(llvm_ir_file, assembly_file):
    args = ["llc", llvm_ir_file, "-o", assembly_file]
    return subprocess.run(args)


def compile_llvm_ir_to_assembly(llvm_ir_dir: str,
                                llc_assembly_dir: str,
                                llvm_ir_suffix: str = "ll",
                                assembly_suffix: str = "s",
                                nproc=1):
    """
    Compile the ANG benchmark.
    """
    llvm_ir_files_relative = preprocessing_utils.get_all_files_with_extension(
        llvm_ir_dir, llvm_ir_suffix)
    llvm_ir_files_abs = [
        os.path.join(llvm_ir_dir, llvm_ir_file)
        for llvm_ir_file in llvm_ir_files_relative if llvm_ir_file != ""
    ]
    assembly_files_abs = [
        os.path.join(llc_assembly_dir,
                     llvm_ir_file[:-len(llvm_ir_suffix)] + assembly_suffix)
        for llvm_ir_file in llvm_ir_files_relative if llvm_ir_file != ""
    ]
    print(f"Get number of files: {len(llvm_ir_files_abs)}")
    # Get all directories
    all_dirs = set()
    for llvm_ir_file in tqdm.tqdm(llvm_ir_files_abs):
        all_dirs.add(os.path.dirname(llvm_ir_file))
    print(all_dirs)
    for d in tqdm.tqdm(all_dirs):
        d = d.replace(llvm_ir_dir, llc_assembly_dir)
        Path(d).mkdir(parents=True, exist_ok=True)

    args = [(llvm_ir_file, assembly_file) for llvm_ir_file, assembly_file in
            zip(llvm_ir_files_abs, assembly_files_abs)]

    with Pool(nproc) as p:
        outputs_lists = p.starmap(compile_llc, args)
        for output in outputs_lists:
            if output.returncode != 0:
                print(output)


def merge_ir_assembly_to_json(llvm_ir_dir: str,
                         llc_assembly_dir: str,
                         output_path: str,
                        llvm_ir_suffix: str = "ll",
                        assembly_suffix: str = "s"):
    """Merge the llvm ir and assembly code from seperated files to a json file.
    """
    llvm_ir_dirs, llvm_ir_files = preprocessing_utils.list_subdirectories_and_files(
        llvm_ir_dir)
    llvm_ir_files_abs = [r for r in tqdm.tqdm(llvm_ir_files) if r != "" and r.endswith(llvm_ir_suffix)]
    assembly_files_abs = [ir_file[:-len(llvm_ir_suffix)] + assembly_suffix for ir_file in tqdm.tqdm(llvm_ir_files_abs)]
    # replace the llvm_ir_dir with llc_assembly_dir
    assembly_files_abs = [ir_file.replace(llvm_ir_dir, llc_assembly_dir) for ir_file in tqdm.tqdm(assembly_files_abs)]
    
    all_data = []
    for llvm_ir_file, assembly_file in tqdm.tqdm(zip(llvm_ir_files_abs,
                                                      assembly_files_abs)):
        # Merge if successfully compiled
        if os.path.exists(assembly_file) and os.path.exists(llvm_ir_file):
            llvm_ir = Path(llvm_ir_file).read_text()
            assembly = Path(assembly_file).read_text()
            if len(llvm_ir) > 0 and len(assembly) > 0:
                all_data.append({
                    "llvm_ir": llvm_ir,
                    "assembly": assembly,
                    "file": llvm_ir_file,
                })
    logger.info(f"Total number of records: {len(all_data)}")
    json.dump(all_data, open(output_path, 'w'), indent=4, sort_keys=True, separators=(',', ': '))
    logger.info(f"Save records to: {output_path}")



def tokenize_text(text: str, tokenizer: transformers.AutoTokenizer) -> list:
    return tokenizer.encode(text)


bb_count_binary: str = "/data0/xiachunwei/Projects/llm4compiler/src/cpp/build/count_llvm_ir_bb"

def count_llvm_ir_bb(llvm_ir_file: str):
    if not isinstance(llvm_ir_file, str) or not os.path.exists(llvm_ir_file) and os.path.isfile(llvm_ir_file):
        print(f"llvm_ir_file is not a string: {llvm_ir_file}")
    try:
        cmd_out = subprocess.run([bb_count_binary, llvm_ir_file], stdout=subprocess.PIPE)
        # Example of output: "{"func_name": "BusFault_Handler" ,"bbcount":2,"bb_list_size": [1,1]}"
        llvm_ir_bb_count = json.loads(cmd_out.stdout.decode("utf-8"))
        return llvm_ir_bb_count
    except :
        logger.info(f"Error Counting bb for: {llvm_ir_file} output: {cmd_out.stdout}")
    return {}


def analyze_or_tokenize_ir_assembly(ir_assembly_path: str,
                         dst_json_path: str,
                         dst_pickle_path: str,
                         tokenizer: AutoTokenizer,
                         nproc=40,
                         save_tokenized: bool = False):
    """Pairwise IR and assembly code from two directories.

    """
    all_data = json.load(open(ir_assembly_path, 'r'))
    # Tokenize the data
    ir_args = [(record["llvm_ir"], tokenizer) for record in all_data]
    assembly_args = [(record["assembly"], tokenizer) for record in all_data]
    ir_file_args = [(record["file"], ) for record in all_data]
    with Pool(nproc) as p:
        ir_encoded = p.starmap(tokenize_text, ir_args)
        assembly_encoded = p.starmap(tokenize_text, assembly_args)
        bb_count = p.starmap(count_llvm_ir_bb, ir_file_args)
    logger.info("Finished analyze all records and")
    # Pack to dict
    new_all_data = []
    tokenized_record_all = []
    for record, ir, assembly, count in zip(all_data, ir_encoded, assembly_encoded, bb_count):
        # Filter out the record with empty ir or assembly
        if not "bbcount" in count.keys():
            continue
        record["llvm_ir_length"] = len(ir)
        record["assembly_length"] = len(assembly)
        record["llvm_ir_bb_count"] = count
        new_all_data.append(record)
        if tokenizer is not None and save_tokenized:
            tokenized_record = record.copy()
            tokenized_record["llvm_ir"] = ir
            tokenized_record["assembly"] = assembly
            tokenized_record_all.append(tokenized_record)

    logger.info("Now saving to disk")
    # Save the data
    json.dump(new_all_data, open(dst_json_path, 'w'))
    if tokenizer is not None and save_tokenized:
        pickle.dump(tokenized_record_all, open(dst_pickle_path, 'wb'))


def main(
    ir_assembly_path = "/data0/xiachunwei/Dataset/decompilation-dataset/AnghaBench-llvm-ir-llc-assembly-O2-content.json",
    dst_json_path = "/data0/xiachunwei/Dataset/decompilation-dataset/AnghaBench-llvm-ir-llc-assembly-O2.json",
    dst_pickle_path = "/data0/xiachunwei/Dataset/decompilation-dataset/AnghaBench-llvm-ir-llc-assembly-O2.pkl",
    model_path = "/data0/xiachunwei/Dataset/CodeLlama-7b-hf",
    nproc=40
):
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    analyze_or_tokenize_ir_assembly(ir_assembly_path, dst_json_path, dst_pickle_path, tokenizer, nproc=nproc, save_tokenized=False)


def filter_record(record: dict, length_threshold: int = 0, bb_count_threshold: int = 0, average_bb_size: float = 1) -> bool:
    """Filter the llvm ir record, return true if the record is valid else return false.

    Args:
        record (dict): The record to be filtered.
        length_threshold (int, optional): The length threshold. Defaults to 0.
        bb_count_threshold (int, optional): The basic block count threshold. Defaults to 0.
        average_bb_size (float, optional): The average basic block size. Defaults to 0.

    Returns:
        bool: True if the record is valid else return false.
    """
    if length_threshold > 0 and record["length"] > length_threshold:
        return False
    if bb_count_threshold > 0:
        bb_count = int(record["llvm_ir_bb_count"]["bbcount"])
        bb_list_size = list(record["llvm_ir_bb_count"]["bb_list_size"])
        if bb_count < bb_count_threshold:
            return False
        if sum(bb_list_size) / bb_count <= average_bb_size:
            return False
    return True


def limit_Length_and_bb_count_and_draw_llvm_ir_and_assembly(llvm_ir_assembly_data_path: str, 
    json_less: str, length_threshold: int = 1024 * 6, bb_count_threshold = 0, average_bb_size: float = 1.0):
    data = json.load(open(llvm_ir_assembly_data_path, 'r'))
    assembly_length_list, llvm_ir_length_list, total_length_list = [], [], []
    bb_length_list = []

    json_less_data = []
    for record in tqdm.tqdm(data):
        assembly_length_list.append(record["assembly_length"])
        llvm_ir_length_list.append(record["llvm_ir_length"])
        total_length = record["assembly_length"] + record["llvm_ir_length"]
        total_length_list.append(total_length)
        bb_length_list.append(record["llvm_ir_bb_count"]["bbcount"])
        record["length"] = record["assembly_length"] + record["llvm_ir_length"]
        if filter_record(record, length_threshold, bb_count_threshold, average_bb_size):
           json_less_data.append(record)
    print("Start sorting the data based on length")
    json_less_data.sort(key=lambda record: record["length"])
    print(f"Dump total number of records: {len(json_less_data)}")
    draw.draw_length_distribution(assembly_length_list, "AnghaBench-llc-assembly_length_distribution.png")
    draw.draw_length_distribution(llvm_ir_length_list, "AnghaBench-llvm_ir_length_distribution.png")
    draw.draw_length_distribution(total_length_list, "AnghaBench-total_length_distribution.png")
    draw.draw_length_distribution(bb_length_list, "AnghaBench-bb_length_list_distribution.png")
    json.dump(json_less_data, open(json_less, 'w'), indent=4, sort_keys=True, separators=(',', ': '))


def prepare_chat(in_json: str, out_json: str):
    in_data =json.load(open(in_json, 'r'))
    code_length = []
    out_data = []
    for record in in_data:
        out_record = {
                "file": record["file"],
                "input": record["assembly"],
                "output": record["llvm_ir"],
                "instruction": "decompile the x86 assembly to llvm ir"
                }
        code_length.append(record["length"])
        out_data.append(out_record)
    pickle.dump(code_length, open("AnghaBench-total_length_list.pkl", 'wb'))
    json.dump(out_data, open(out_json, 'w'), indent=4, sort_keys=True, separators=(',', ': ') )


if __name__=="__main__":
    # compile_llvm_ir_to_assembly(
    #  "/data0/xiachunwei/Dataset/decompilation-dataset/AnghaBench-ll-O2",
    # "/data0/xiachunwei/Dataset/decompilation-dataset/AnghaBench-llc-assembly-O2",
    # nproc=32)
    # dataset_dir = "/data0/xiachunwei/Dataset/decompilation-dataset/"
    # merge_ir_assembly_to_json(llvm_ir_dir = os.path.join(dataset_dir, "AnghaBench-ll-O2"),
    #     llc_assembly_dir = os.path.join(dataset_dir, "AnghaBench-llc-assembly-O2"),
    #     output_path = os.path.join(dataset_dir, "AnghaBench-llvm-ir-llc-assembly-O2-content.json"))
    ir_assembly_path = "/data0/xiachunwei/Dataset/decompilation-dataset/AnghaBench-llvm-ir-llc-assembly-O2-content.json"
    dst_json_path = "/data0/xiachunwei/Dataset/decompilation-dataset/AnghaBench-llvm-ir-llc-assembly-O2.json"
    dst_pickle_path = "/data0/xiachunwei/Dataset/decompilation-dataset/AnghaBench-llvm-ir-llc-assembly-O2.pkl"
    model_path = "/data0/xiachunwei/Dataset/CodeLlama-7b-hf"
    nproc=80
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    analyze_or_tokenize_ir_assembly(ir_assembly_path, dst_json_path, dst_pickle_path, tokenizer, nproc=nproc, save_tokenized=False)
    # fire.Fire(main)
    # limit_Length_and_bb_count_and_draw_llvm_ir_and_assembly(
    #     "/data0/xiachunwei/Dataset/decompilation-dataset/AnghaBench-llvm-ir-llc-assembly-O2.json",
    #     "/data0/xiachunwei/Dataset/decompilation-dataset/AnghaBench-llvm-ir-llc-assembly-O2-seq_length-4K-bbcount-2.json",
    #     length_threshold=1024*4,
    #     bb_count_threshold=2)
    # prepare_chat("/data0/xiachunwei/Dataset/decompilation-dataset/AnghaBench-llvm-ir-llc-assembly-O2-seq_length-4K-bbcount-2.json",
    #         "/data0/xiachunwei/Dataset/decompilation-dataset/AnghaBench-llvm-ir-llc-assembly-O2-seq_length-4K_bbcount-2_chat.json")

