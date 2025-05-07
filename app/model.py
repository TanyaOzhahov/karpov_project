import mlflow
import mlflow.sklearn
import os

_model = None

def load_latest_model(force_reload=False):
    global _model
    if _model is None or force_reload:
        model_uri = f"models:/tatjana-ozhahovskaja-wbb7567-mlops-project-model@prod"
        _model = mlflow.pyfunc.load_model(model_uri)
    return _model

def predict(model, passwords: list[str]) -> list[float]:
    import pandas as pd
    df = pd.DataFrame({"Password": passwords})
    preds = model.predict(df)
    return preds.tolist()
