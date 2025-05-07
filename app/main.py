from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from app.model import predict, load_latest_model
from app.trainer import run_training_pipeline

app = FastAPI()
model = None  # глобально, но инициализируется позже

class PredictRequest(BaseModel):
    Password: list[str]

class TriggerRequest(BaseModel):
    data_url: str

@app.post("/predict")
def predict_passwords(request: PredictRequest):
    global model
    if model is None:
        model = load_latest_model()
        if model is None:
            return {"error": "Model not loaded yet. Call /trigger first."}
    return {"Times": predict(model, request.Password)}

@app.post("/trigger")
def trigger_training(request: TriggerRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(run_training_pipeline, request.data_url)
    return {"status": "pipeline triggered"}

