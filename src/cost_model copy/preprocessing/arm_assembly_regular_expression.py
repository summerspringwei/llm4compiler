
import re
from typing import List

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


def get_extract_hex16(inst: str) -> List[str]:
    """
    Match exactly the hex16 number
    """
    exact_hex16_pattern = r'^[0-9a-fA-F]{16}$'
    return re.findall(exact_hex16_pattern, inst)


def get_hex16(inst: str) -> List[str]:
    """
    Get all hex16 numbers in the instruction
    """
    hex16_pattern = r'[0-9a-fA-F]{16}'
    return re.findall(hex16_pattern, inst)

def get_label(inst: str) -> List[str]:
    """
    Get all labels in the instruction
    match `<main>:` and return the `main`    
    """
    label_pattern = r'<[a-zA-Z_.][a-zA-Z0-9_]*>:'
    label_name_pattern = r'[a-zA-Z_.][a-zA-Z0-9_]*'
    result = []
    for label in re.findall(label_pattern, inst):
        result += re.findall(label_name_pattern, label)
    return result

