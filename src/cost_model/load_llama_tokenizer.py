

from transformers import CodeLlamaTokenizer
# llama_tokenizer_dir="/home/xiachunwei/Dataset/CodeLlama-7b-hf"
llama_tokenizer_dir="llama_data/cbench_wo_line_no/merged_tokenizer_hf/"
llama_tokenizer = CodeLlamaTokenizer.from_pretrained(llama_tokenizer_dir)

vocab = llama_tokenizer.get_vocab()
print(vocab)
