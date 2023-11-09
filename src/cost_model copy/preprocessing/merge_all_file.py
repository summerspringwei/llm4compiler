import os
from typing import List

from custom_logging import get_custom_logger
logger = get_custom_logger()
"""This file is created to merge all the assembly files into one file for each benchmark.
Then we feed the merged assembly file to the sentencepiece model to build the vocabulary.

This file depends on the following files to produce the results:
`python3 extract_pass_seq_by_latency.py`
"""

def merge_all_assembly(root_dir: str, output_file_path: str):
    if os.path.exists(output_file_path):
        os.remove(output_file_path)
    
    items = os.listdir(root_dir)
    # Filter out only the directories from the list
    for item in items:
        benchmark_dir = os.path.join(root_dir, item)
        if not os.path.isdir(benchmark_dir):
            continue
        ir_dir = os.path.join(benchmark_dir, "random", "samples")
        all_ir_directories = os.listdir(ir_dir)
        for dir in all_ir_directories:
            file_path = os.path.join(ir_dir, dir, "a.s")
            if os.path.exists(file_path):
                os.system(f"cat {file_path} >> {output_file_path}")
        logger.info(f"Processed benchmark {item}")



def split_assembly_by_chunk_size(lines: List[str], chunk_size=2000):
    results = [lines[i * chunk_size: (i + 1) * chunk_size] for i in range((len(lines) + chunk_size - 1) // chunk_size )] 
    return results


def pack_assmbly_lines_to_json(lines: List[str]):
    new_lines = []
    # First filter out the empty lines
    for line in lines:
        line = line.strip()
        if line == "":
            continue
        new_lines.append(line)
    line_str = "\\n".join(new_lines)
    
    return '{"input_ids": "' + line_str + '"}'


def prepare_pre_train_assembly_dataset(root_dir: str, output_file_path: str, chunk_size=2000):
    if os.path.exists(output_file_path):
        os.remove(output_file_path)
    fout = open(output_file_path, "w")

    items = os.listdir(root_dir)
    # Filter out only the directories from the list
    # Loop over all benchmarks
    for item in items:
        benchmark_dir = os.path.join(root_dir, item)
        if not os.path.isdir(benchmark_dir):
            continue
        ir_dir = os.path.join(benchmark_dir, "random", "samples")
        all_ir_directories = os.listdir(ir_dir)
        # Loop over all samples
        for c_dir in all_ir_directories:
            file_path = os.path.join(ir_dir, c_dir, "a.s")
            if not os.path.exists(file_path):
                logger.warning(f"File {file_path} does not exist")
                continue
            with open(file_path, "r") as f:
                lines = f.readlines()
                outputs = []
                if len(lines) > chunk_size:
                    outputs = split_assembly_by_chunk_size(lines, chunk_size)
                else:
                    outputs = [lines]
                for out in outputs:
                    fout.write(pack_assmbly_lines_to_json(out)+"\n")
            # Firstly we only process one sample for each benchmark
            break

        logger.info(f"Processed benchmark {item}")
    
    fout.flush()
    fout.close()

if __name__ == "__main__":
    folder_path = "/home/xiachunwei/Projects/llm4compiler/src/cost_model/"
    # merge_all_assembly(os.path.join(folder_path, "cBench"), os.path.join(folder_path, "all_assembly.txt"))
    prepare_pre_train_assembly_dataset(os.path.join(folder_path, "cBench"), os.path.join(folder_path, "all_benchmark_pre_train.txt"))
