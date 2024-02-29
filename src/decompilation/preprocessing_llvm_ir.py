
import os
import subprocess
import json
import ndjson
from multiprocessing import Pool

from common import custom_logging
from common import preprocessing_utils

logger = custom_logging.get_custom_logger()


"""
We compile the C source file to object file and dump the llvm ir with the following command:
clang -mllvm -print-after=pre-isel-intrinsic-lowering -c test.c -O2 > test.ll

The test.ll contains some debug information and we need to remove them.
The format of the test.ll is like:


extr_xprepare.c_xdl_init_classifier.c:19:22: warning: incompatible redeclaration of lib
xxxx
*** IR Dump After Pre-ISel Intrinsic Lowering (pre-isel-intrinsic-lowering) ***
source_filename = "/data0/xiachunwei/Dataset/decompilation-dataset/AnghaBench/git/xdiff/extr_xprepare.c_xdl_init_classifier.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

%struct.TYPE_3__ = type { i64, i32, i32, i64, i64, i32, ptr, ptr }
xxx

xxx warning generated.
"""

def extract_llvm_ir_inplace(llvm_ir_file: str) -> str:
    """Read an llvm ir file and preprocess it.

    Args:
        llvm_ir_file (str): Path to the llvm ir file.
    
    Returns:
        output_lines (str): lines to write to.
    """
    start, end = 0, -1
    failed = False
    with open(llvm_ir_file, 'r') as f:
        lines = f.readlines()
        for idx, line in zip(range(len(lines)), lines):
            if "*** IR Dump After" in line or line.startswith("source_filename"):
                start = idx+1
            if "warning generated." in line or "warnings generated." in line:
                end = idx
                break
            if "error generated" in line or "errors generated" in line:
                failed = True
                break
    
    if end == -1 and not failed:
        end = len(lines)
    if start >=0 and end >= 0:
        lines = lines[start:end]
        with open(llvm_ir_file, 'w') as f:
            f.writelines(lines)
        result = "".join(lines)
        return result
    else:
        os.remove(llvm_ir_file)
        return None


def extract_llvm_ir_dir(llvm_ir_dir: str, nproc: int = 32) -> None:
    llvm_ir_files_relative = preprocessing_utils.get_all_files_with_extension(llvm_ir_dir, "ll")
    llvm_ir_files_abs = [os.path.join(llvm_ir_dir, llvm_ir_file) for llvm_ir_file in llvm_ir_files_relative if llvm_ir_file != ""]
    # Preprocess each llvm ir file
    with Pool(nproc) as p:
        output_lines = p.map(extract_llvm_ir_inplace, llvm_ir_files_abs)
    logger.info("Preprocessed {} llvm ir files are written to {}".format(len(output_lines), llvm_ir_dir + ".json"))


def test(file_path: str):
    output_lines = extract_llvm_ir_inplace(file_path)
    print(output_lines)


if __name__ == "__main__":
    # test("/data0/xiachunwei/Dataset/decompilation-dataset/AnghaBench-ll-O2/git/xdiff/extr_xutils.c_xdl_recmatch.ll")
    extract_llvm_ir_dir("/data0/xiachunwei/Dataset/decompilation-dataset/AnghaBench-ll-O2/")
