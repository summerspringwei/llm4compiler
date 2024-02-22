import os
from pathlib import Path
from multiprocessing import Pool
import json
import pickle

import fire
import transformers
from transformers import AutoTokenizer

from common import custom_logging
from common import preprocessing_utils


logger = custom_logging.get_custom_logger()


def tokenize_text(text: str, tokenizer: transformers.AutoTokenizer) -> list:
    return tokenizer.encode(text)


def pairwise_ir_assembly(llvm_ir_dir: str,
                         llc_assembly_dir: str,
                         dst_json_path: str,
                         dst_pickle_path: str,
                         tokenizer: AutoTokenizer,
                         llvm_ir_suffix: str = "ll",
                         assembly_suffix: str = "s",
                         nproc=1):
    """Pairwise IR and assembly code from two directories.

    """
    llvm_ir_files_relative = preprocessing_utils.get_all_files_with_extension(
        llvm_ir_dir, llvm_ir_suffix)
    llvm_ir_files_relative = [r for r in llvm_ir_files_relative if r != ""]
    llvm_ir_files_abs = [
        os.path.join(llvm_ir_dir, llvm_ir_file)
        for llvm_ir_file in llvm_ir_files_relative if llvm_ir_file != ""
    ]
    assembly_files_abs = [
        os.path.join(llc_assembly_dir,
                     llvm_ir_file[:-len(llvm_ir_suffix)] + assembly_suffix)
        for llvm_ir_file in llvm_ir_files_relative if llvm_ir_file != ""
    ]
    all_data = []
    for llvm_ir_file, assembly_file, llvm_ir_r in zip(llvm_ir_files_abs,
                                                      assembly_files_abs,
                                                      llvm_ir_files_relative):
        # Merge if successfully compiled
        if os.path.exists(assembly_file) and os.path.exists(llvm_ir_dir):
            llvm_ir = Path(llvm_ir_file).read_text()
            assembly = Path(assembly_file).read_text()
            if len(llvm_ir) > 0 and len(assembly) > 0:
                all_data.append({
                    "llvm_ir": llvm_ir,
                    "assembly": assembly,
                    "file:": llvm_ir_r
                })
    logger.info(f"Total number of records: {len(all_data)}")
    # Tokenize the data
    ir_args = [(record["llvm_ir"], tokenizer) for record in all_data]
    assembly_args = [(record["assembly"], tokenizer) for record in all_data]
    with Pool(nproc) as p:
        ir_encoded = p.starmap(tokenize_text, ir_args)
        assembly_encoded = p.starmap(tokenize_text, assembly_args)
    logger.info("Finished tokenize all records")
    # Pack to dict
    new_all_data = []
    tokenized_record_all = []
    for record, ir, assembly in zip(all_data, ir_encoded, assembly_encoded):
        record["llvm_ir_length"] = len(ir)
        record["assembly_length"] = len(assembly)
        new_all_data.append(record)
        tokenized_record = record.copy()
        tokenized_record["llvm_ir"] = ir
        tokenized_record["assembly"] = assembly
        tokenized_record_all.append(tokenized_record)
    logger.info("Now saving to disk")
    # Save the data
    json.dump(new_all_data, open(dst_json_path, 'w'))
    pickle.dump(tokenized_record_all, open(dst_pickle_path, 'wb'))



def main(
    llvm_ir_dir = "/data0/xiachunwei/Dataset/decompilation-dataset/AnghaBench-ll-O2",
    llc_assembly_dir = "/data0/xiachunwei/Dataset/decompilation-dataset/AnghaBench-llc-assembly-O2",
    dst_json_path = "/data0/xiachunwei/Dataset/decompilation-dataset/AnghaBench-llvm-ir-llc-assembly-O2.json",
    dst_pickle_path = "/data0/xiachunwei/Dataset/decompilation-dataset/AnghaBench-llvm-ir-llc-assembly-O2.pkl",
    model_path = "/data0/xiachunwei/Dataset/CodeLlama-7b-hf",
    nproc=40
):
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    pairwise_ir_assembly(llvm_ir_dir, llc_assembly_dir, dst_json_path, dst_pickle_path, tokenizer, nproc=nproc)





if __name__=="__main__":
    fire.Fire(main)
