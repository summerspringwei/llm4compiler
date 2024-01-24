

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

I provide a Makefile thus we don't need to write python code to compile.
We can let `make` to manage the compilation/decompilation.
Simply run the following code:

```shell
cd src/decompilation
# modify the compilation flags in Makefile, and then
make -j -i
```
The reason why we pass `-i` to make is that clang failed to compile some files.
Note, currently we use `clang-15`.

After compilation, the assembly files (end with `*.s`) will be saved to a folder with the same structure as the source code.
Then we need to pre-processing the assembly code.
```shell
python3 decompilation/preprocessing_assembly.py --dataset_dir path/to/decompilation-dataset/ --dir_name AnghaBench-assembly-g-O2
```

### Training

#### Convert weight
Before starting training, we need to convert the huggingface weight (model.layers.x.xxx) to the GPT weights (transformers.h.x.xxx).

```shell
cd path/to/TinyLlama
python3 scripts/convert_hf_checkpoint.py --checkpoint_dir /workspace/Dataset/TinyLlama-1.1B-step-50K-105b/ --model_name tiny_LLaMA_1b  --dtype float32
```
Then we can initialize the model with the huggingface weights and continue the training process.

Note, the converted weights will be saved in the save to `checkpoint_dir` and named as `lit_model.pth`.

#### Continue Pretrain

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

#### Convert pretrained model to huggingface format
After pretrain the model, we need to convert the trained GPT model weight (transformers.h.x.xxx) to huggingface (model.layers.x.xxx) so that we can use the transformers `pipeline` to load and run model.
```shell
python scripts/convert_lit_checkpoint.py --out_dir out/tinyllama_1b/ --checkpoint_name iter-024000-ckpt.pth --model_name tiny_LLaMA_1b
```

Note, the converted weights will be saved to the `out_dir` and named as `iter-024000-ckpt.bin`.

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
