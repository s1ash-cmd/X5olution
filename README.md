# NER RuBERT Service

Данный репозиторий содержит сервис для распознавания именованных сущностей (NER) на основе модели **DeepPavlov/rubert-base-cased**.  
Проект включает обучение модели, офлайн-инференс по CSV и REST API на базе FastAPI.

---

## Описание возможностей
- Обучение модели NER (файл `rubert-base-cased.py`)
- Тестирование и инференс по CSV (файл `test_isp.py`)
- REST API для предсказаний (файлы `main.py` и `model.py`)
- Поддержка BIO-разметки:
  ```
  O, B-TYPE, I-TYPE, B-BRAND, I-BRAND, B-VOLUME, I-VOLUME, B-PERCENT, I-PERCENT
  ```

---

## Структура проекта
```
.
├── main.py              # FastAPI сервис
├── model.py             # Класс NERModel
├── rubert-base-cased.py # Скрипт обучения
├── test_isp.py          # Офлайн-инференс
├── requirements.txt     # Зависимости
├── train.csv            # Датасет для обучения (ожидается)
├── submission.csv       # Датасет для валидации (ожидается)
├── test.csv             # Датасет для теста (опционально)
└── ner_rubert/          # Папка с обученной моделью
```

---

## Установка

```bash
git clone <repo_url>
cd <repo_name>

python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate    # Windows

pip install -r requirements.txt
```

---

## Обучение модели

Файл `rubert-base-cased.py` обучает модель на данных `train.csv` и `submission.csv`.

```bash
python rubert-base-cased.py
```

После обучения модель будет сохранена в папке `ner_rubert/`.

---

## Офлайн-инференс

Для предсказаний по `test.csv` используется:

```bash
python test_isp.py
```

Результаты сохраняются в файл `test_predictions.csv`.

---

## Запуск API

```bash
uvicorn main:app --reload --port 8000
```

Документация:
- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- OpenAPI JSON: [http://localhost:8000/openapi.json](http://localhost:8000/openapi.json)

---

## Пример использования API

**POST** `/api/predict`

Входные данные:
```json
{
  "input": "Газированный напиток Coca-Cola 0.5%"
}
```

Результат:
```json
[
  { "start_index": 18, "end_index": 27, "entity": "B-BRAND" },
  { "start_index": 29, "end_index": 33, "entity": "B-VOLUME" },
  { "start_index": 34, "end_index": 37, "entity": "B-PERCENT" }
]
```

---

## Возможные ошибки

1. `ModuleNotFoundError: No module named 'torch'` — необходимо установить совместимую версию PyTorch.
2. Все предсказания равны `O` — проверьте корректность разметки в `annotation` в CSV.
3. Использование GPU — добавьте перенос модели и тензоров на `cuda` в `model.py`.

---

## Лицензия

- Базовая модель: [DeepPavlov/rubert-base-cased](https://huggingface.co/DeepPavlov/rubert-base-cased)
- Код в данном репозитории распространяется по лицензии MIT (или укажите иную при необходимости).
