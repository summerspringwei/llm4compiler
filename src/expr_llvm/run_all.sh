#!/bin/bash

clang -c test.c -o test.o

llvm-objdump --no-show-raw-insn -D ./test.o > test.s

clang -mllvm -print-after=pre-isel-intrinsic-lowering -c test.c -O0 > test.ll 2>&1

llc test.ll -o test_llvm_ir.s

as -o test_llvm_ir.o test_llvm_ir.s