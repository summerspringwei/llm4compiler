from typing import List
import re
from build_vocabulary import add_vocabulary

def preprocessing_with_addr(arm_assembly: List[str], strip_line_number = True)->str:
    """
    Preprocessing the arm assembly code

    We mainly replace lines in the assembly code with "\n"
    and return a string
    """
    add_vocabulary(arm_assembly, add_line_addr = False)
    # Insert tokens 
    num_token = 0
    processed_str = ""
    for line in arm_assembly:
        if line.find("/home")>=0:
            continue
        com = line.strip().replace("\t", " ").split(" ")
        com = [x for x in com if x != '']
        if len(com) == 0:
            continue 
        if strip_line_number:
            if re.match(r'^[0-9a-fA-F]{6}:$', com[0]):
                com = com[1:]
        line = " ".join(com)
        processed_str += (line + "\n")
        num_token += (len(com) + 1)
    return processed_str
