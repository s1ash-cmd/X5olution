import math
import ast
import pandas as pd
import numpy as np


from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    DataCollatorForTokenClassification,
    TrainingArguments,
)
import transformers as tf
from seqeval.metrics import accuracy_score, f1_score, classification_report

# ==== 1. Данные ====
train_df = pd.read_csv("train.csv", sep=";")
test_df  = pd.read_csv("submission.csv", sep=";")

# ==== 2. Метки ====
label_list = ["O", "B-TYPE", "I-TYPE", "B-BRAND", "I-BRAND", "B-VOLUME", "I-VOLUME", "B-PERCENT", "I-PERCENT"]
label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for l, i in label2id.items()}

def normalize_tag(tag: str) -> str:
    if tag is None:
        return "O"
    t = str(tag).strip().upper()
    if t in ("0", "O"):
        return "O"
    TYPES = {"TYPE", "BRAND", "VOLUME", "PERCENT"}
    if t in TYPES:
        return f"B-{t}"
    if "-" in t:
        prefix, tail = t.split("-", 1)
        prefix = prefix.strip().upper()
        tail   = tail.strip().upper()
        if tail in TYPES and prefix in {"B", "I"}:
            return f"{prefix}-{tail}"
    return "O"

def make_bio(sentence, annotation):
    tokens = str(sentence).split()
    labels = ["O"] * len(tokens)
    if annotation is None or (isinstance(annotation, float) and np.isnan(annotation)):
        spans = []
    elif isinstance(annotation, (list, tuple)):
        spans = annotation
    elif isinstance(annotation, str):
        ann_str = annotation.strip()
        if ann_str == "" or ann_str.upper() == "NAN":
            spans = []
        else:
            try:
                spans = ast.literal_eval(ann_str)
            except Exception:
                spans = []
    else:
        spans = []
    for item in spans:
        if not (isinstance(item, (list, tuple)) and len(item) >= 3):
            continue
        start, end, raw_tag = item[0], item[1], item[2]
        tag = normalize_tag(raw_tag)
        if tag == "O":
            continue
        char_idx = 0
        first_token = True
        for i, tok in enumerate(tokens):
            token_start = char_idx
            token_end   = char_idx + len(tok)
            if token_end > start and token_start < end:
                if first_token:
                    labels[i] = tag
                    first_token = False
                else:
                    ent_type = tag.split("-", 1)[-1]
                    labels[i] = f"I-{ent_type}"
            char_idx = token_end + 1
    return tokens, labels

def convert_df(df: pd.DataFrame) -> Dataset:
    new_data = {"tokens": [], "ner_tags": []}
    for _, row in df.iterrows():
        tokens, labels = make_bio(row["sample"], row["annotation"])
        label_ids = [label2id.get(l, label2id["O"]) for l in labels]
        new_data["tokens"].append(tokens)
        new_data["ner_tags"].append(label_ids)
    return Dataset.from_dict(new_data)

train_dataset = convert_df(train_df)
test_dataset  = convert_df(test_df)

dataset = DatasetDict({
    "train": train_dataset,
    "validation": test_dataset
})

# ==== 5. Токенайзер ====
model_name = "DeepPavlov/rubert-base-cased"
tokenizer  = AutoTokenizer.from_pretrained(model_name)
tokenizer.model_max_length = 512

def tokenize_and_align_labels(examples):
    tokenized = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=512
    )
    labels = []
    for i, word_labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        prev = None
        label_ids = []
        for w_id in word_ids:
            if w_id is None:
                label_ids.append(-100)
            elif w_id != prev:
                label_ids.append(word_labels[w_id])
            else:
                label_ids.append(-100)
            prev = w_id
        labels.append(label_ids)
    tokenized["labels"] = labels
    return tokenized

tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True, remove_columns=[])


# ==== 6. Модель ====
model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)

# ==== 7. Тренировка (минимально совместимый набор аргументов) ====
args = TrainingArguments(
    output_dir="./ner_rubert",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    report_to=[],   # <-- отключает wandb
)

data_collator = DataCollatorForTokenClassification(tokenizer)

def compute_metrics(p):
    preds, labels = p
    preds = np.argmax(preds, axis=2)
    true_labels = [[label_list[l] for l in label if l != -100] for label in labels]
    true_preds  = [[label_list[p_] for (p_, l) in zip(pred, label) if l != -100]
                   for pred, label in zip(preds, labels)]
    return {
        "accuracy": accuracy_score(true_labels, true_preds),
        "f1": f1_score(true_labels, true_preds),
    }

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],  # понадобится для ручной оценки
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

import transformers as tf
print("transformers version:", tf.__version__)

trainer.train()                 # обучение без auto-eval во время эпох
metrics = trainer.evaluate()    # ручная валидация после обучения
print(metrics)
trainer.save_model("./ner_rubert")        # сохраняет pytorch_model.bin + конфиг
tokenizer.save_pretrained("./ner_rubert") # сохраняет токенизатор
