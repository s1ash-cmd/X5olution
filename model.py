from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

class NERModel:
    def __init__(self, model_path="./ner_rubert"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model.eval()

    def predict(self, text: str):
        if not text.strip():
            return []

        tokens = self.tokenizer(text, return_offsets_mapping=True, return_tensors="pt", truncation=True)
        offsets = tokens.pop("offset_mapping")

        with torch.no_grad():
            outputs = self.model(**tokens).logits

        preds = torch.argmax(outputs, dim=-1)[0].tolist()
        offsets = offsets[0].tolist()

        # объединяем подсловные токены
        results = []
        current_entity = None
        for (start, end), label_id in zip(offsets, preds):
            if start == end:
                continue
            label = self.model.config.id2label[label_id]
            if label == "O":
                if current_entity:
                    results.append(current_entity)
                    current_entity = None
                continue
            if current_entity and current_entity["entity"] == label and current_entity["end_index"] == start:
                current_entity["end_index"] = end
            else:
                if current_entity:
                    results.append(current_entity)
                current_entity = {"start_index": start, "end_index": end, "entity": label}
        if current_entity:
            results.append(current_entity)

        return results

ner_model = NERModel()
