from typing import List
import re
from build_vocabulary import add_vocabulary
from custom_logging import get_custom_logger
logger = get_custom_logger()

def get_sections_after_text(arm_assembly: List[str])->List[str]:
    """
    Get the sections after .text
    """
    start_idx = 0
    for line, idx in zip(arm_assembly, range(len(arm_assembly))):
        line = line.strip()
        if line == "Disassembly of section .text:":
            start_idx = idx
            break
    if start_idx == 0:
        logger.error("Cannot find the .text section in the disassembly code")
    return arm_assembly[start_idx:]


def preprocessing_with_addr(
        arm_assembly: List[str], 
        strip_line_number = True, 
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",)->str:
    """
    Preprocessing the arm assembly code

    We mainly replace lines in the assembly code with "\n"
    and return a string
    """
    # Here we strip library functions (typically .plt) in the disassembly code
    # and only keep the code after the .text section
    arm_assembly = get_sections_after_text(arm_assembly)
    # add_vocabulary(arm_assembly, add_line_addr = False)
    # Insert tokens 
    num_token, num_sentence = 0, 0
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
        num_sentence += 1
    logger.info(f"Number of tokens: {num_token}, number of sentences: {num_sentence}")
    return processed_str
