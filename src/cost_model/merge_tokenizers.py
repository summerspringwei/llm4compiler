
"""
This file is downloaded from https://github.com/ymcui/Chinese-LLaMA-Alpaca/
We need to merge our own assembly with CodeLLaMa's tokenizers.
"""

import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]="python"
from transformers import CodeLlamaTokenizer
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
import sentencepiece as spm
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--llama_tokenizer_dir', default=None, type=str, required=True)
parser.add_argument('--ir_model_file', default='./cbench_sp.model', type=str)
args = parser.parse_args()

llama_tokenizer_dir = args.llama_tokenizer_dir
cbench_ir_sp_model_file = args.ir_model_file

# load
llama_tokenizer = CodeLlamaTokenizer.from_pretrained(llama_tokenizer_dir)
cbench_ir_sp_model = spm.SentencePieceProcessor()
cbench_ir_sp_model.Load(cbench_ir_sp_model_file)

llama_spm = sp_pb2_model.ModelProto()
llama_spm.ParseFromString(llama_tokenizer.sp_model.serialized_model_proto())
cbench_ir_spm = sp_pb2_model.ModelProto()
cbench_ir_spm.ParseFromString(cbench_ir_sp_model.serialized_model_proto())

# print number of tokens
print(len(llama_tokenizer),len(cbench_ir_sp_model))
print(llama_tokenizer.all_special_tokens)
print(llama_tokenizer.all_special_ids)
print(llama_tokenizer.special_tokens_map)

## Add Chinese tokens to LLaMA tokenizer
llama_spm_tokens_set=set(p.piece for p in llama_spm.pieces)
print(len(llama_spm_tokens_set))
print(f"Before:{len(llama_spm_tokens_set)}")
for p in cbench_ir_spm.pieces:
    piece = p.piece
    if piece not in llama_spm_tokens_set:
        new_p = sp_pb2_model.ModelProto().SentencePiece()
        new_p.piece = piece
        new_p.score = 0
        llama_spm.pieces.append(new_p)
print(f"New model pieces: {len(llama_spm.pieces)}")

## Save
output_sp_dir = 'merged_tokenizer_sp'
output_hf_dir = 'merged_tokenizer_hf' # the path to save Chinese-LLaMA tokenizer
os.makedirs(output_sp_dir,exist_ok=True)
with open(output_sp_dir+'/cbench_ir_llama.model', 'wb') as f:
    f.write(llama_spm.SerializeToString())
tokenizer = CodeLlamaTokenizer(vocab_file=output_sp_dir+'/cbench_ir_llama.model')

tokenizer.save_pretrained(output_hf_dir)
print(f"Chinese-LLaMA tokenizer has been saved to {output_hf_dir}")

# Test
llama_tokenizer = CodeLlamaTokenizer.from_pretrained(llama_tokenizer_dir)
chinese_llama_tokenizer = CodeLlamaTokenizer.from_pretrained(output_hf_dir)
print(tokenizer.all_special_tokens)
print(tokenizer.all_special_ids)
print(tokenizer.special_tokens_map)
text='''
<dequeue>:
stp	x29, x30, [sp, #-0x10]!
adrp	x9, 0x412000 <puts@@GLIBC_2.17+0x412000>
ldr	x8, [x9, #0xa0]
mov	x29, sp
The primary use of LLaMA is research on large language models, including'''
print("Test text:\n",text)
print(f"Tokenized by LLaMA tokenizer:{llama_tokenizer.tokenize(text)}")
print(f"Tokenized by cBench-IR-LLaMA tokenizer:{chinese_llama_tokenizer.tokenize(text)}")
