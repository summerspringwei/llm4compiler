import os
import re
from typing import List

from custom_logging import get_custom_logger
logger = get_custom_logger()

global arm64_vocabulary
arm64_vocabulary = set()

static_key_words = {"Disassembly", "of", "section", ".init", "$x", "+", "-"}
arm64_vocabulary.intersection_update(static_key_words)

def add_vocabulary(arm_assembly: List[str], add_line_addr = False)->str:
    """
    Add vocabulary from arm assembly code
    
    """
    
    full_hex16_pattern = r'^[0-9a-fA-F]{16}$'
    code_addr_mark_pattern = r'^[0-9a-fA-F]{6}:$'
    content_of_register_pattern = r'^\[[w|x][0-9]+\]'
    register_pattern = r'[w|x][0-9]+'
    hex_number_pattern = r'0x[0-9a-fA-F]+'
    instance_number_pattern = r'^#[+|-]?0x[0-9a-fA-F]+'
    def_addr_label_pattern = r'^<[a-zA-Z_.][a-zA-Z0-9_]*>:$' # e.g. <main1>:
    addr_label_pattern = r'[a-zA-Z_.][a-zA-Z0-9_]*'
    # jump_to_label_pattern = r'^<[[a-zA-Z_.][a-zA-Z0-9_]*][[@[a-zA-Z_.][a-zA-Z0-9_]*]|[[+|-]0x[0-9a-fA-F]+]]?>$'
    jump_to_label_pattern = r'^<[[a-zA-Z_.][a-zA-Z0-9_]*][@[a-zA-Z_.][a-zA-Z0-9_]*]?[[+|-]0x[0-9a-fA-F]+]?>$'
    addr_label_plus_instance_number_pattern = r'^<[a-zA-Z_][a-zA-Z0-9_]*[\+|-]0x[0-9a-fA-F]+>$'
    register_addr_plus_number_pattern = r'^\[[[w|x][0-9]+|sp|pc], #[+|-]?0x[0-9a-fA-F]+\]!?$'
    call_func_lib_pattern = r'^<[a-zA-Z_][a-zA-Z0-9_]*@[a-zA-Z_][a-zA-Z0-9_]>:?'
    function_name_pattern = r'[a-zA-Z_][a-zA-Z0-9_]*'
    before_num_elements = len(arm64_vocabulary)
    for line in arm_assembly:
        if line.find("/home")>=0:
            continue
        com = line.strip().replace("\t", " ").split(" ")
        com = [x for x in com if x != '']
        new_token = []
        for token in com:
            token = token.strip()
            if token in static_key_words:
                continue
            # if token is a full hex16 number, e.g. 0000000000400808
            if re.match(full_hex16_pattern, token):
                if add_line_addr:
                    arm64_vocabulary.add(token)
                else:
                    continue
            # if token is a code address mark, e.g. 400814
            elif re.match(code_addr_mark_pattern, token):
                if add_line_addr:
                    arm64_vocabulary.add(token)
                else:
                    continue
            # if token is a register address, e.g. [w0]
            elif re.findall(content_of_register_pattern, token):
                all_registers = re.findall(register_pattern, token)
                if len(all_registers) > 0:
                    arm64_vocabulary.add(all_registers[0])
            # if token is content of a register plus number, e.g. [w0, #0x10]
            elif re.findall(register_addr_plus_number_pattern, token):
                # In this case, we should consider the instance number
                match_result = re.findall(register_addr_plus_number_pattern, token)
                if len(match_result) > 0:
                    instance_number_lists = re.findall(instance_number_pattern, match_result[0])
                    if len(instance_number_lists) > 0:
                        instance_number = instance_number_lists[0]
                        if instance_number.find("!")>=0:
                            logger.warn("Error: instance number not found in token: ", token)
                        if instance_number.find(']')>=0:
                            logger.warn("Error: instance number extract wrong: ", token, instance_number)
                        arm64_vocabulary.add(instance_number)
            # if token is a instance number, e.g. #0x10
            elif re.findall(instance_number_pattern, token):
                arm64_vocabulary.add(token)
            # Match all addr label, e.g. <main1>,
            elif re.findall(def_addr_label_pattern, token):
                def_addr_label = re.findall(addr_label_pattern, token)
                if len(def_addr_label) > 0:
                    arm64_vocabulary.add(def_addr_label[0])
            # if token is a call/jump to label,e.g. <main1>, <__libc_start_main@libc> or <bfopen+0x44>
            elif re.match(jump_to_label_pattern, token):
                def_addr_label = re.findall(addr_label_pattern, token)
                for label in def_addr_label:
                    arm64_vocabulary.add(label)
                instance_number = re.findall(hex_number_pattern, token)
                if len(instance_number) > 0:
                    arm64_vocabulary.add(instance_number[0])
            elif re.findall(addr_label_plus_instance_number_pattern, token):
                # split the label name and instance number
                label = re.findall(function_name_pattern, token)
                if len(label) > 0:
                    arm64_vocabulary.add(label[0])
                else :
                    logger.warn("Error: label not found in token: ", token)
                instance_number = re.findall(hex_number_pattern, token)
                if len(instance_number) > 0:
                    arm64_vocabulary.add(instance_number[0])
                else:
                    logger.warn("Error: instance number not found in token: ", token)
            # Other pattern set to a single token
            else:
                if token.find("]")>=0:
                    token = token.replace("]", "")
                if token.find(",")>=0:
                    token = token.replace(",", "")
                arm64_vocabulary.add(token)
    after_num_elements = len(arm64_vocabulary)
    logger.info(f"Add {after_num_elements - before_num_elements} new elements to vocabulary")


def dump_vocabulary(vocabulary_path: str):
    # Last time to clear the vocabulary
    new_arm64_vocabulary = set()
    for token in arm64_vocabulary:
            if token.find("]")>=0:
                token = token.replace("]", "")
            new_arm64_vocabulary.add(token)
    logger.info(f"clear vocabulary from {len(new_arm64_vocabulary)} to {len(new_arm64_vocabulary)} elements")
    with open(vocabulary_path, 'w') as f:
        for token in new_arm64_vocabulary:
            f.write(token+"\n")
    logger.info(f"Dump vocabulary to {vocabulary_path} with {len(new_arm64_vocabulary)} elements")
