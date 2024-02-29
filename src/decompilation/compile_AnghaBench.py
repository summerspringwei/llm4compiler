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


if __name__ == "__main__":
    compile_AnghaBench(
        "/home/xiachunwei/Dataset/decompilation-dataset/AnghaBench/",
        "/home/xiachunwei/Dataset/decompilation-dataset/AnghaBench/",
        "/home/xiachunwei/Dataset/decompilation-dataset/AnghaBench-obj-g-O2/",
        compile_args=['-g', '-O2'],
        nproc=32)

