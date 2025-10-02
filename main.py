from fastapi import FastAPI
from pydantic import BaseModel
from model import ner_model
import asyncio
from concurrent.futures import ThreadPoolExecutor

app = FastAPI()
executor = ThreadPoolExecutor(max_workers=2)

class InputData(BaseModel):
    input: str

@app.post("/api/predict")
async def predict(data: InputData):
    text = data.input.strip()
    if not text:
        return []

    # Запуск синхронной модели в отдельном потоке
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(executor, ner_model.predict, text)
    return results
