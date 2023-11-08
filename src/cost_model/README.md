# Using Large Language Model to Build Cost Model for CPU Programs


## Prepare data from cBench benchmarking results

### Data format
For detailed data format, refer to [this link](https://github.com/summerspringwei/debug_new_pass_manager/blob/main/cbench_dataset/README.md) for explaination.

We mainly prepare the data as follows:
1. Sort tuning records by latency;
2. Draw histogram figure and bars to see the distrbution of latency;
3. We sample `n=100` records from all the records by steps;
4. Disassembly the binary to assembly file with `llvm-objdump`;
4. Choose the performance gap large than `performance_eps=0.3` as pairs of training record;
5. Save the training records to json file named `llm_training_record.json`

### How to generate training data:
```shell
python3 extract_pass_seq_by_latency.py
```

### Merge all the assembly code to one file
#### We also use this scripts to prepare the pre-train dataset for all arm assembly
We should use tokenizer to decide how to split the assembly.
```shell
python3 merge_all_file.py
```
This would merge all the assembly file into one large txt file named `all_assembly.txt`.

### Use google's sentencepiece to build the vocabulary
```shell
bash scripts/run_sentencepiece.sh all_assembly.txt
```
This will produce two files:
`xxx.model` and `xxx.vocab`.

### Merge the vocabulary
Then we need to merge the vocabulary produced by sentencepiece with CodeLamma's vocabulary
```shell
python3 merge_tokenizers.py
```

### Test the length of the tokenizer
```shell
python3 run_demo_codellama.py
```

### Train sft with LoRA
```shell
bash scripts/run_sft.sh
```

### Run the llm serving and client
Run the following command to start the llm server:
```shell
bash scripts/run_llm_serving.sh
```
Run the server to send a request and get a response:
```shell
python3 llm_serving/llm_client.py
```


## Large Language Model for Performance Cost Model
This sub-directory is developed for building performance code by using Large Language Model (LLM).

Run codellama demo:
```shell
python3 run_demo_codellama.py
```

### Tokenizer
We need to build our own vocabulary for LLVM IR or assembly.
Now we have implemented a vocabulary build `build_vocabulary.py` 
by matching regular expressions to extract registers, label names et.al.


Whether to add immediate operand to the vocabulary is a key problem.
The reason is that if we do not put them into vocabulary,
each digital number will be regarded as a token and the total sequence length for
a program will be too long to be trained.

We also consider to use machine learning tools to automatically build vocabulary.
The most important one is the [sentencepiece](https://github.com/google/sentencepiece).
After building the vocabulary, refer to [Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca/wiki/Training-Details) to merge the vocabulary with code llama's vocabulary.

### The choice of language

Assembly or LLVM IR, this is a question.
The good thing to use assembly code is that It is more concise and can refect the underline hardware behavious.
However, there are too many instructions that contains instant numbers and it's hard for the the LLM to trace the address. 
The good thing to use LLVM IR is that it is more close to the program and it's still using labels to mark the entry points. Thus maybe it's better for LLM to trace the jump/call instructions. The disadvantage of using LLVM IR is the each instruction has more tokens (7-10) than assembly code (3-6).

We need also use LLVM IR to train the cost model.

## Explore OpenAI's API

### Tokenizer
We can use OpenAI's tokenizer to count how many tokens in our training record.
refer to [tiktoken](https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken) for more info.

### Embeddings
We can also get the embeddings for our text by using OpenAI's [Embeddings](https://platform.openai.com/docs/guides/embeddings/what-are-embeddings) API.


## TODO
Use grpc so we can load the model and tokenizer once,
and run multiple inference. (Done)

## TODO
* Read the Tokenizer code
* Download the Dataset and run the simple demo in `finetune-llama.py`.
* Refer to this [link](https://github.com/liguodongiot/llm-action/blob/main/train/peft/clm/peft_lora_clm.ipynb) and [this](https://towardsdatascience.com/fine-tune-your-own-llama-2-model-in-a-colab-notebook-df9823a04a32) to fine-tune.
