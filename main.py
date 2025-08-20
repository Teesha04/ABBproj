from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from enum import Enum
from model import XGB

app = FastAPI()

class MLModels(str, Enum):
    XGB = "xgboost"
    LIGHTGBM = "lightgbm"

class TrainPayload(BaseModel):
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    model_name: MLModels
    file_name: str

class ModelResponse(BaseModel):
    error: str | None
    accuracy: float
    precision: float
    recall: float
    f1: float

@app.post("/train-model", response_model=ModelResponse)
async def train_and_return_metrics(data: TrainPayload):
    if data.model_name == MLModels.XGB:
        x = XGB(data.train_start, data.train_end, data.test_start, data.test_end, data.file_name)
        x.train_and_predict()
        metrics = x.get_metrics()
        if metrics['error'] != None:
            raise HTTPException(status_code=500, detail=metrics.error)
        else:
            return metrics