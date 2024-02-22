import numpy as np

from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset
import evaluate


books = load_dataset("opus_books", "en-fr")
books = books["train"].train_test_split(train_size=1024, test_size=128)
print(books["train"].shape, books["train"][0])

checkpoint = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

source_lang = "en"
target_lang = "fr"
prefix = "translate English to French: "

def preprocess_function(examples):
    inputs = [prefix + example[source_lang] for example in examples["translation"]]
    targets = [prefix + example[target_lang] for example in examples["translation"]]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=1024, truncation=True)
    print(type(tokenizer))
    return model_inputs


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decode_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decode_pres, decode_labels = postprocess_text(decoded_preds, decode_labels)
    metric = evaluate.load("sacrebleu")
    result = metric.compute(predictions=decoded_preds, references=decode_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}

    return result


# print(books["train"][:1])
# model_inputs = preprocess_function(books["train"][:8])
# print(model_inputs)
# # The len(attention_mask) == len(input_ids)
# print(len(model_inputs["input_ids"][0]), len(model_inputs["labels"][0]), len(model_inputs["attention_mask"][0]))
# input_sentence = tokenizer.decode(model_inputs["input_ids"][0])
# print(input_sentence)
# labels_sentence = tokenizer.decode(model_inputs["labels"][0])
# print(labels_sentence)


tokenized_books = books.map(preprocess_function, batched=True)

# def convert_to_features(example_batch):
#     input_ids = example_batch["input_ids"]
#     attention_mask = example_batch["attention_mask"]
#     labels = example_batch["labels"]

#     result = []
#     for id, mask, label in zip(input_ids, attention_mask, labels):
#         result.append({"input_ids": id, "attention_mask": mask, "labels": label})

#     return result

data_collator = DataCollatorForSeq2Seq(tokenizer, model=checkpoint)

# results = convert_to_features(model_inputs)
# print(type(results))
# # list_model_inputs = [inputs.data for inputs in model_inputs]
# results = data_collator(results)
# print(results)
# print(results["input_ids"].shape, results["attention_mask"].shape, results["labels"].shape)


training_args = Seq2SeqTrainingArguments(
    output_dir="./T5-small-translation",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
    predict_with_generate=True,
    logging_dir="./logs",
    bf16=True,
    half_precision_backend="auto"
)

model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_books["train"],
    eval_dataset=tokenized_books["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()

