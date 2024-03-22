import os
import subprocess
import json
import glob
import tqdm

# def load_json(file_path: str) -> str:
#     with open(file_path, "r") as f:
#         json_obj = json.load(f)
#         for obj in json_obj:
#             print(obj)
#         return json_obj

def load_json(file_path: str):
    with open(file_path, "r") as f:
        lines = f.readlines()
        json_objs = [json.loads(line) for line in lines]
        print(type(json_objs[0]))
        return json_objs
    return None


def get_all_files_with_extension(dir_path: str, extension: str) -> list:
    """Get all files with a specific extension in a directory.

    Args:
        dir_path (str): Path to the directory.
        extension (str): File extension.

    Returns:
        file_paths (list): List of file paths.
    """
    if not os.path.exists(dir_path):
        logger.error("Directory {} does not exist.".format(dir_path))
        return
    if extension == None or extension == "":
        logger.error("Extension cannot be empty.")
        return
    file_paths = []
    for sub_dir in tqdm.tqdm(os.listdir(dir_path)):
        cmd_process  = subprocess.Popen(["find", ".", "-name", "*." + extension], cwd=dir_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        cmd_stdout, cmd_stderr = cmd_process.communicate()
        sub_file_paths = cmd_stdout.decode("utf-8").split("\n")
        sub_file_paths = [file_path for file_path in sub_file_paths if file_path != ""]
        file_paths.extend(sub_file_paths)
    return file_paths


def list_subdirectories_and_files(directory):
    all_dirs, all_files = [], []
    for root, dirs, files in os.walk(directory):
        # Print all subdirectories
        for subdir in dirs:
            all_dirs.append(os.path.join(root, subdir))
        
        # Print all files
        # print("Files:")
        for file in files:
            all_files.append(os.path.join(root, file))
    return all_dirs, all_files