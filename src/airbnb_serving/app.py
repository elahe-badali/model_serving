import os
from contextlib import asynccontextmanager
from pathlib import Path

import mlflow
import mlflow.sklearn
from fastapi import FastAPI

from airbnb_serving.predictor import predict_batch, predict_single
from airbnb_serving.schema import ListingFeatures, PredictionResponse

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[2]


load_dotenv(BASE_DIR / ".env")

model = None
model_run_id = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, model_run_id

    model_run_id = os.environ["MODEL_RUN_ID"]

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    username = os.environ.get("MLFLOW_TRACKING_USERNAME")
    password = os.environ.get("MLFLOW_TRACKING_PASSWORD")

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    if username:
        os.environ["MLFLOW_TRACKING_USERNAME"] = username

    if password:
        os.environ["MLFLOW_TRACKING_PASSWORD"] = password

    model_uri = f"runs:/{model_run_id}/sklearn_pipeline_model"
    model = mlflow.sklearn.load_model(model_uri)

    yield

    model = None
    model_run_id = None


app = FastAPI(
    title="Airbnb Demand Prediction API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_run_id": str(model_run_id),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(features: ListingFeatures):
    return predict_single(features, model, str(model_run_id))


@app.post("/predict/batch", response_model=list[PredictionResponse])
def predict_batch_endpoint(features_list: list[ListingFeatures]):
    return predict_batch(features_list, model, str(model_run_id))
