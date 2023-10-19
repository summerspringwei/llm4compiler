import os

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


if __name__ == "__main__":
    folder_path = "/home/xiachunwei/Projects/llm4compiler/src/cost_model/"
    merge_all_assembly(os.path.join(folder_path, "cBench"), os.path.join(folder_path, "all_assembly.txt"))
