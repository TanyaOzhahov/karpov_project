import mlflow
import pandas as pd
from sklearn.linear_model import LinearRegression
from app.utils import load_data_from_url, validate_data
from app.model import load_latest_model

def train_model(df: pd.DataFrame):
    X = df["Password"].apply(len).values.reshape(-1, 1)  # Простейший фиче-инжиниринг
    y = df["Times"]
    model = LinearRegression()
    model.fit(X, y)
    return model

def run_training_pipeline(data_url: str):
    try:
        df = load_data_from_url(data_url)
    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    if not validate_data(df):
        print("Data validation failed.")
        return

    model = train_model(df)

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:8080"))
    mlflow.set_experiment("mlops_project")

    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Регистрируем модель
        result = mlflow.register_model(
            model_uri=f"runs:/{run.info.run_id}/model",
            name="tatjana-ozhahovskaja-wbb7567-mlops-project-model"
        )

        # Устанавливаем alias "prod"
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        client.set_registered_model_alias(
            name="tatjana-ozhahovskaja-wbb7567-mlops-project-model",
            alias="prod",
            version=result.version
        )

    print("Training completed and model registered.")
