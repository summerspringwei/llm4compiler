import os
import glob
from pathlib import Path
from multiprocessing import Pool
import subprocess

from common import custom_logging
from common import preprocessing_utils

logger = custom_logging.get_custom_logger()


def compile_src2obj(src_path: str, dest_path: str, compile_args: str):
    args = ["clang", *compile_args, "-c", src_path, "-o", dest_path]
    return subprocess.run(args)


def compile_AnghaBench(work_dir: str,
                       root_dir: str,
                       dest_dir: str,
                       compile_args=['-g', '-O0'],
                       nproc=1):
    """
    Compile the ANG benchmark.
    """
    files = glob.glob(work_dir, recursive=True)
    src_files = []
    # Get all the source files recursively
    for file in files:
        if os.path.isdir(file):
            dest_path = Path(file.replace(root_dir, dest_dir))
            dest_path.mkdir(parents=True, exist_ok=True)
            print("d", dest_path)
            compile_AnghaBench(file + "/*", root_dir, dest_dir, compile_args)
        else:
            src_files.append(file)
    args = [(src, Path(src.replace(
        root_dir,
        dest_dir)).with_suffix(".o").absolute().as_posix(), compile_args)
            for src in src_files]
    with Pool(nproc) as p:
        outputs_lists = p.starmap(compile_src2obj, args)
        # for output in outputs_lists:
        #     if output.returncode != 0:
        #         print(output)


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

    # Get all directories
    all_dirs = set()
    for llvm_ir_file in llvm_ir_files_abs:
        all_dirs.add(os.path.dirname(llvm_ir_file))
    print(all_dirs)
    for d in all_dirs:
        d = d.replace(llvm_ir_dir, llc_assembly_dir)
        Path(d).mkdir(parents=True, exist_ok=True)

    args = [(llvm_ir_file, assembly_file) for llvm_ir_file, assembly_file in
            zip(llvm_ir_files_abs, assembly_files_abs)]

    with Pool(nproc) as p:
        outputs_lists = p.starmap(compile_llc, args)
        for output in outputs_lists:
            if output.returncode != 0:
                print(output)


if __name__ == "__main__":
    # compile_AnghaBench(
    #     "/home/xiachunwei/Dataset/decompilation-dataset/AnghaBench/",
    #     "/home/xiachunwei/Dataset/decompilation-dataset/AnghaBench/",
    #     "/home/xiachunwei/Dataset/decompilation-dataset/AnghaBench-obj-g-O2/",
    #     compile_args=['-g', '-O2'],
    #     nproc=32)
    compile_llvm_ir_to_assembly(
        "/data0/xiachunwei/Dataset/decompilation-dataset/AnghaBench-ll-O2",
        "/data0/xiachunwei/Dataset/decompilation-dataset/AnghaBench-llc-assembly-O2",
        nproc=32)
