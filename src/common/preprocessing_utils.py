import os
import subprocess
import json


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
    cmd_process  = subprocess.Popen(["find", ".", "-name", "*." + extension], cwd=dir_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    cmd_stdout, cmd_stderr = cmd_process.communicate()
    file_paths = cmd_stdout.decode("utf-8").split("\n")
    file_paths = [file_path for file_path in file_paths if file_path != ""]
    return file_paths
