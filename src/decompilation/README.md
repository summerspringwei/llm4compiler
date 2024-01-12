

## Large Language Model for De-Compilation

### Data Pre-Processing

#### Prepareing the AnghaBench

The benchmark can be found at [AnghaBench](https://github.com/brenocfg/AnghaBench).
In this benchmark, each source file contains one function and the function's dependency declaration.
Thus we can get the standalone assembly for one function.

 - [ ] Maybe we should use llvm to generate our own dataset like AnghaBench.

Firstly, we compile the source code `*.c` to object file `*.o` using `clang`. e.g.
```shell
clang -c -g -O0 *.c -o *.o
```
Then we get the corresponding assembly code by using `llvm-objdum`:
```shell
llvm-objdump -s -d --no-addresses --no-show-raw-insn *o
```
Now I think address is quit import so I would rather keep the addresses in the assembly code

Get the corresponding 

Run the following command to get all the object file and assembly file:
```shell
bash run_compile_anghabench.sh
```


## Tips

Dump the LLVM IR after certain pass:
set [ref](https://lists.llvm.org/pipermail/llvm-dev/2016-June/100986.html)
```shell
clang -c -g -mllvm -print-after-all *.o
```
We can get the dumped passes and corresponding llvm IR by grep `*** IR Dump After Pass Name (pass) ***`.
We can also get the llvm IR for a specific pass:
```shell
clang -c -g -mllvm -print-after=verify *.o
```
We can get the llvm IR before the instruction selection phase.
Maybe it's helpful for the training.