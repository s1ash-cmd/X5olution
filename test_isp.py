import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer

# ==== 1. Метки ====
label_list = ["O", "B-TYPE", "I-TYPE", "B-BRAND", "I-BRAND",
              "B-VOLUME", "I-VOLUME", "B-PERCENT", "I-PERCENT"]
label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for l, i in label2id.items()}

# ==== 2. Загрузка токенизатора и модели ====
model_name = "DeepPavlov/rubert-base-cased"
tokenizer  = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(
    "./ner_rubert/checkpoint-10221",   # путь к последнему чекпоинту
    id2label=id2label,
    label2id=label2id
)

trainer = Trainer(model=model, tokenizer=tokenizer)

# ==== 3. Подготовка теста ====
test_df = pd.read_csv("test.csv", sep=";")

def convert_test_df(df: pd.DataFrame) -> Dataset:
    new_data = {"tokens": [], "sample": []}
    for _, row in df.iterrows():
        tokens = str(row["sample"]).split()
        new_data["tokens"].append(tokens)
        new_data["sample"].append(str(row["sample"]))
    return Dataset.from_dict(new_data)


test_dataset = convert_test_df(test_df)

def tokenize(examples):
    return tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=512
)


tokenized_test = test_dataset.map(tokenize, batched=True)

# ==== 4. Предсказания ====
predictions, _, _ = trainer.predict(tokenized_test)
pred_ids = np.argmax(predictions, axis=2)

# ==== 5. Формируем интервалы (start, end, label) ====.

import re

def word_starts_in_text(text: str, tokens: list[str]) -> list[int]:
    """Находит глобальные индексы начала каждого токена в исходной строке."""
    starts = []
    pos = 0
    for tok in tokens:
        m = re.search(re.escape(tok), text[pos:])
        if m is None:
            m = re.search(re.escape(tok), text)
            if m is None:
                s = pos
            else:
                s = m.start()
        else:
            s = pos + m.start()
        starts.append(s)
        pos = s + len(tok)
    return starts


results = []
for i, tokens in enumerate(test_dataset["tokens"]):
    sample_text = test_dataset["sample"][i]

    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_offsets_mapping=True
    )

    word_ids = encoding.word_ids()
    pred_labels_per_word = {}
    seen = set()
    for j, w_id in enumerate(word_ids):
        if w_id is None:
            continue
        if w_id not in seen:
            seen.add(w_id)
            pred_label_id = pred_ids[i][j]
            pred_labels_per_word[w_id] = id2label[int(pred_label_id)]

    # глобальные начала слов в исходной строке
    starts = word_starts_in_text(sample_text, tokens)

    spans = []
    for w_id, label in sorted(pred_labels_per_word.items()):
        start = starts[w_id]
        end = start + len(tokens[w_id])  # целиком слово, а не первый сабтокен
        spans.append((start, end, label))

    results.append(str(spans))

# сохраняем под нужным именем столбца
out_df = pd.DataFrame({
    "sample": test_df["sample"],
    "annotation": results
})

out_df.to_csv("test_predictions.csv", sep=";", index=False, encoding="utf-8")
print("✅ Предсказания сохранены в test_predictions_spans.csv")
