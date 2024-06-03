

"""
This script is used for extract the assembly information from the llvm-mc AsmParser output.
Recall how we compile an C source code to object file:
```bash
    clang -S -emit-llvm -O2 test.c -o test.ll
    llc -O2 test.ll
    llvm-mc --filetype=obj test.s -o test.o
```
By passing `--show-inst-operands` llvm-mc, we can obtain the parsed instructions with operands like this:
```
foo.s:29:2: note: parsed instruction: [callq, Memory: ModeSize=64,Scale=1,Disp=printf]
        callq   printf@PLT
        ^
foo.s:30:2: note: parsed instruction: [movss, Memory: ModeSize=64,BaseReg=rbp,Scale=1,Disp=-4, Reg:xmm0]
        movss   -4(%rbp), %xmm0                 # xmm0 = mem[0],zero,zero,zero
        ^
```
"""


import re


def extract_function_call(call_inst: str):
    com = call_inst.split(", ")
    if len(com) == 2 and com[0].startswith("call"):
        # Extract the called function: callq, Memory: ModeSize=64,Scale=1,Disp=printf
        match = re.search(r'Disp=(.*)', com[1])
        if match:
            return match.group(1)
    return None


def extract_asm_info(f):
    called_functions = []
    for line in f:
        match = re.search(r'parsed instruction: \[(.*?)\]', line)
        if match:
            func_name = extract_function_call(match.group(1))
            if func_name is not None:
                called_functions.append(func_name)
    return called_functions


def test():
    content = """"
    foo.s:29:2: note: parsed instruction: [callq, Memory: ModeSize=64,Scale=1,Disp=printf]
        callq   printf@PLT
        ^
    foo.s:30:2: note: parsed instruction: [movss, Memory: ModeSize=64,BaseReg=rbp,Scale=1,Disp=-4, Reg:xmm0]
            movss   -4(%rbp), %xmm0                 # xmm0 = mem[0],zero,zero,zero
            ^
    """
    content = content.split("\n")
    print(extract_asm_info(content))


if __name__=="__main__":
    test()
