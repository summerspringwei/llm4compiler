from build_vocabulary import arm64_vocabulary, dump_vocabulary
from preprocessing_arm_assembly import preprocessing_with_addr
import os
import hashlib
import json
from typing import List

from custom_logging import get_custom_logger
logger = get_custom_logger()


class Record:
    def __init__(self, pass_seq, latency):
        self.pass_seq = str(pass_seq)
        self.latency = float(latency)
        self.assembly: str = None
        self.llvm_ir = None

    def __str__(self) -> str:
        return f"pass seq: {self.pass_seq}, latency: {self.latency}"

    def get_pass_seq_hash(self) -> str:
        return hashlib.md5(self.pass_seq.encode('utf-8')).hexdigest()


def read_jiayu_result_json(path: str) -> List[Record]:
    record_list = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            com = line.strip()[1:-1].split(',')
            record_list.append(Record(com[0][1:-1], com[1]))
    return record_list


def sort_record_list(record_list: List[Record]) -> List[Record]:
    return sorted(record_list, key=lambda x: x.latency)


def sample_from_sorted_record_list(record_list: List[Record], sample_num: int) -> List[Record]:
    sample_list = []
    sample_interval = len(record_list) // sample_num
    for i in range(sample_num):
        sample_list.append(record_list[i * sample_interval])
    return sample_list


def draw_record_latency_distribution(record_list: List[Record], output_path: str, benchmark_name="bench"):
    import matplotlib.pyplot as plt
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    latency_list = [record.latency for record in record_list]
    plt.hist(latency_list, bins=100)
    plt.xlabel('Latency')
    plt.ylabel('Number of samples')
    plt.title(f'{benchmark_name} Latency distribution')
    plt.savefig(output_path + ".pdf")
    plt.savefig(output_path + ".png")
    plt.close()


def draw_record_latency_bars(record_list: List[Record], output_path: str, benchmark_name="bench"):
    import matplotlib.pyplot as plt
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    latency_list = [record.latency for record in record_list]
    idx_list = [i for i in range(len(latency_list))]
    plt.bar(idx_list, latency_list)
    plt.xlabel('index')
    plt.ylabel('Latency')
    plt.title(f'{benchmark_name} Sampled latency')
    plt.savefig(output_path + ".pdf")
    plt.savefig(output_path + ".png")
    plt.close()


def disassembly_binary(binary_path: str, output_dir: str):
    return os.system(f"llvm-objdump -d --no-addresses --no-show-raw-insn {binary_path} > {output_dir}")


def get_record_list_assembly(record_list: List[Record],
                             from_all_binary_dir: str, to_all_binary_dir: str) -> List[Record]:
    new_record_list = []
    for record in record_list:
        pass_seq_hash = record.get_pass_seq_hash()
        from_binary_dir = os.path.join(
            from_all_binary_dir, "IR-"+pass_seq_hash)
        to_binary_dir = os.path.join(to_all_binary_dir, "IR-"+pass_seq_hash)
        if not os.path.exists(to_binary_dir):
            os.makedirs(to_binary_dir)
        binary_path = os.path.join(from_binary_dir, "a.out")
        assembly_path = os.path.join(to_binary_dir, "a.s")
        status = disassembly_binary(binary_path, assembly_path)
        if status != 0:
            logger.warn(f"Disassembly binary {binary_path} failed")
        # Preprocessing the assembly code
        with open(assembly_path, 'r') as f:
            assembly = f.readlines()
            record.assembly = preprocessing_with_addr(assembly)
        new_record_list.append(record)

    return new_record_list


def get_record_list_assembly(record_list: List[Record],
                             from_all_binary_dir: str, to_all_binary_dir: str) -> List[Record]:
    new_record_list = []
    for record in record_list:
        pass_seq_hash = record.get_pass_seq_hash()
        from_binary_dir = os.path.join(
            from_all_binary_dir, "IR-"+pass_seq_hash)
        to_binary_dir = os.path.join(to_all_binary_dir, "IR-"+pass_seq_hash)
        if not os.path.exists(to_binary_dir):
            os.makedirs(to_binary_dir)
        binary_path = os.path.join(from_binary_dir, "a.out")
        assembly_path = os.path.join(to_binary_dir, "a.s")
        status = disassembly_binary(binary_path, assembly_path)
        if status != 0:
            logger.warn(f"Disassembly binary {binary_path} failed")
        # Preprocessing the assembly code
        with open(assembly_path, 'r') as f:
            assembly = f.readlines()
            record.assembly = preprocessing_with_addr(assembly)
        new_record_list.append(record)

    return new_record_list


def construct_llm_training_record(record_list: List[Record],
                                  max_num_record: int, output_path: str,
                                  performance_eps=0.05):
    """Construct training record for llm with json format

    Note: the record_list must sort by latency from low to high
    The format is as follows:

    llm_training_record = {
        "instruction:": "compare the performance of the two arm64 assembly code",
        "code1:": "arm64 assembly code 1",
        "code2:": "arm64 assembly code 2",
        "output": "code1 is faster/slower than code2"
    }
    """
    low_idx, high_idx = 0, len(record_list) - 1
    num_record = 0
    meta_f = open(os.path.join(os.path.dirname(output_path), "meta.txt"), 'w')
    with open(os.path.join(output_path), 'w') as f:
        while low_idx < high_idx and num_record < max_num_record:
            low_record = record_list[low_idx]
            high_record = record_list[high_idx]
            real_eps = (high_record.latency - low_record.latency) / \
                high_record.latency

            if real_eps > performance_eps:
                (code_1, code_2) = (low_record, high_record) if num_record % 2 == 0 else (
                    high_record, low_record)
                output = "code1 is faster than code2" if num_record % 2 == 0 else "code1 is slower than code2"
                llm_training_record = {
                    "instruction:": "compare the performance of the two arm64 assembly code",
                    "code1:": code_1.assembly,
                    "code2:": code_2.assembly,
                    "output": output
                }
                meta_record = {
                    "code1_hash": code_1.get_pass_seq_hash(),
                    "code1_latency:": code_1.latency,
                    "code2_hash": code_2.get_pass_seq_hash(),
                    "code2_latency:": code_2.latency
                }
                # Also save metadata about which two records are compared
                meta_f.write(json.dumps(meta_record)+"\n")
                f.write(json.dumps(llm_training_record)+"\n")
                if num_record % 1 == 0:
                    logger.info(f"Construct {num_record} llm training record")
                num_record += 1

            # Update the index
            low_idx += 1
            high_idx -= 1
        print(f"Construct llm training dataset with {num_record} record")


def build_all_benchmark(jiayu_root_dir: str, root_dir: str, sample_sorted_num=100, max_num_record=100):
    items = os.listdir(jiayu_root_dir)
    # Filter out only the directories from the list
    directories = [item for item in items if os.path.isdir(
        os.path.join(jiayu_root_dir, item))]
    for item in items:
        jiayu_benchmark_dir = os.path.join(jiayu_root_dir, item, "random")
        benchmark_dir = os.path.join(root_dir, item, "random")
        if not os.path.exists(benchmark_dir):
            os.makedirs(benchmark_dir)
        print(jiayu_benchmark_dir, benchmark_dir)
        build_one_benchmark(jiayu_benchmark_dir, benchmark_dir, benchmark_name=str(item),
                            sample_sorted_num=sample_sorted_num, max_num_record=max_num_record)
        logger.info(f"Processed benchmark {item}")


def build_one_benchmark(jiayu_benchmark_dir, benchmark_dir, benchmark_name="bench", sample_sorted_num=100, max_num_record=100):
    file_name = "result.json"
    file_path = os.path.join(jiayu_benchmark_dir, file_name)
    # Sort the record list by latency
    record_list = sort_record_list(read_jiayu_result_json(file_path))
    # Draw the latency distribution
    draw_record_latency_distribution(record_list,
                                     os.path.join(benchmark_dir, benchmark_name+"_latency_distribution"), benchmark_name)
    # Draw the sampled latency bars
    sampled_records = sample_from_sorted_record_list(
        record_list, sample_sorted_num)
    draw_record_latency_bars(sampled_records, os.path.join(
        benchmark_dir, benchmark_name+"_latency_bars"), benchmark_name)
    # Disassembly the binary
    get_record_list_assembly(sampled_records,
                             os.path.join(jiayu_benchmark_dir, "samples"),
                             os.path.join(benchmark_dir, "samples"))
    # Construct the training record for llm
    construct_llm_training_record(sampled_records, max_num_record,
                                  os.path.join(benchmark_dir, "llm_training_record.json"), performance_eps=0.03)


if __name__ == "__main__":
    build_one_benchmark("/home/jiayu/result_llvmtunerv3/cBench/automotive_bitcount/random/",
                        "/home/xiachunwei/Projects/llm4compiler/src/cost_model/cBench/automotive_bitcount/random/")
    dump_vocabulary(
        "/home/xiachunwei/Projects/llm4compiler/src/cost_model/cBench/vocabulary.txt")
    # build_all_benchmark("/home/jiayu/result_llvmtunerv3/cBench/",
    #                     "./cBench/", sample_sorted_num=1000, max_num_record=500)
