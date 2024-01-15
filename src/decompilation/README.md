

## Large Language Model for De-Compilation

### Prepare Env

We provide a docker file to build all the dependencies:
```shell
cd docker
docker build -t llm4compiler:latest .
```
It takes about an hour to build the dependencies and may vary based on the performance of machine.
After that, start the docker:
```shell
bash run_docker.sh
```

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
Then we get the corresponding assembly code by using `llvm-objdump`:
```shell
llvm-objdump -s -d --no-addresses --no-show-raw-insn *o
```
Now I think address is quit import so I would rather keep the addresses in the assembly code

Get the corresponding 

Run the following command to get all the object file and assembly file:
```shell
bash run_compile_anghabench.sh
```

### Training

Make sure you have mount the datasets to the docker container.
Then start trainig with the following command:
```shell
cd path/to/TinyLlama

lightning run model \
    --node-rank=0  \
    --main-address=127.0.0.1 \
    --accelerator=cuda \
    --devices=1 \
    --num-nodes=1 \
    pretrain/tinyllama.py --devices 1 \
    --train_data_dir /workspace/Dataset/RedPajama-Data-1T-Sample-Bin  \
    --val_data_dir /workspace/Dataset/RedPajama-Data-1T-Sample-Bin
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
