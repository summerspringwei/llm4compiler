

import os
import subprocess
import json
import ndjson
from multiprocessing import Pool

from common import custom_logging
from common import preprocessing_utils

logger = custom_logging.get_custom_logger()

def preprocess_assembly(asm_file: str) -> str:
    """Read an assembly file and preprocess it.

    Args:
        asm_file (str): Path to the assembly file.
    
    Returns:
        output_lines (str): lines to write to.
    """
    with open(asm_file, 'r') as f:
        lines = f.readlines()
        lines = [line.strip().replace("     ", "") for line in lines[4:]]
        output_lines = "\n".join(lines)
        return output_lines
    
    return None


def pack_assembly_to_json(asm_file: str, relative_path: str) -> None:
    # print(asm_file)
    output_lines = preprocess_assembly(asm_file)
    line = {"assembly_file": relative_path, "text": output_lines}
    json_obj = json.dumps(line)
    return json_obj


def preprocess_assembly_dir(asm_dir: str, output_path: str, nproc: int = 32) -> None:
    asm_files_relative = preprocessing_utils.get_all_files_with_extension(asm_dir, "s")
    asm_files_abs = [os.path.join(asm_dir, asm_file) for asm_file in asm_files_relative if asm_file != ""]

    # Preprocess each assembly file
    with Pool(nproc) as p:
        output_jsons = p.starmap(pack_assembly_to_json, zip(asm_files_abs, asm_files_relative))
    # Write to json file
    ndjson.dump(output_jsons, open(output_path, "w"))
    logger.info("Preprocessed {} assembly files are written to {}".format(len(output_jsons), output_path))


def test():
    asm_file = os.path.join("/home/xiachunwei/Dataset/decompilation-dataset/AnghaBench-Sample-assembly-g-O0/curl/src/extr_slist_wc.c_slist_wc_append.s")
    output_lines = preprocess_assembly(asm_file)


def main(
    dataset_dir = "/home/xiachunwei/Dataset/decompilation-dataset/",
    dir_name = "AnghaBench-assembly-g-O2") -> None:
    asm_dir = os.path.join(dataset_dir, dir_name)
    asm_json_path = os.path.join(dataset_dir, os.path.normpath(dir_name) + ".json")
    preprocess_assembly_dir(asm_dir, asm_json_path)
    # assembly_jsons = load_json(asm_json_path)
    # for assembly_json in assembly_jsons:
    #     print(json.loads(assembly_json)["assembly"])


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(main)
