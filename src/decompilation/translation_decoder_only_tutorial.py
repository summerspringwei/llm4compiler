## Refer to https://huggingface.co/SnypzZz/Llama2-13b-Language-translate for detailed explaination



from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
# article_en = "The head of the United Nations says there is no military solution in Syria"
article_en = """
Recent work demonstrates the potential of multilingual pretraining of creating one model that can be used for various tasks in different languages. Previous work in multilingual pretraining has demonstrated that machine translation systems can be created by finetuning on bitext. In this work, we show that multilingual translation models can be created through multilingual finetuning. Instead of finetuning on one direction, a pretrained model is finetuned on many directions at the same time. Compared to multilingual models trained from scratch, starting from pretrained models incorporates the benefits of large quantities of unlabeled monolingual data, which is particularly important for low resource languages where bitext is not available. We demonstrate that pretrained models can be extended to incorporate additional languages without loss of performance. We double the number of languages in mBART to support multilingual machine translation models of 50 languages. Finally, we create the ML50 benchmark, covering low, mid, and high resource languages, to facilitate reproducible research by standardizing training and evaluation data. On ML50, we demonstrate that multilingual finetuning improves on average 1 BLEU over the strongest baselines (being either multilingual from scratch or bilingual finetuning) while improving 9.3 BLEU on average over bilingual baselines from scratch.
"""
model = MBartForConditionalGeneration.from_pretrained("SnypzZz/Llama2-13b-Language-translate")
tokenizer = MBart50TokenizerFast.from_pretrained("SnypzZz/Llama2-13b-Language-translate", src_lang="en_XX")

model_inputs = tokenizer(article_en, return_tensors="pt")
print(len(model_inputs["input_ids"][0]), len(model_inputs["attention_mask"][0]))
# translate from English to Hindi
# generated_tokens = model.generate(
#     **model_inputs,
#     forced_bos_token_id=tokenizer.lang_code_to_id["hi_IN"]
# )
# tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
# => 'संयुक्त राष्ट्र के नेता कहते हैं कि सीरिया में कोई सैन्य समाधान नहीं है'
print("aaaa")
# translate from English to Chinese
generated_tokens = model.generate(
    **model_inputs,
    forced_bos_token_id=tokenizer.lang_code_to_id["zh_CN"]
)
out = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
print(out)
# => '联合国首脑说,叙利亚没有军事解决办法'
