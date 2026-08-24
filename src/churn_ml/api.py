import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal
from uuid import uuid4

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from .features import SENTIMENT_SCORE_COLUMN

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models/hybrid_churn_bundle.joblib"
MODEL_METADATA_PATH = ROOT / "models/model_metadata.json"
SERVICE_VERSION = os.getenv("SERVICE_VERSION", "0.2.0")
GIT_SHA = os.getenv("GIT_SHA", "unknown")
LOGGER = logging.getLogger("uvicorn.error")

app = FastAPI(title="Hybrid review churn API", version=SERVICE_VERSION)


class CustomerFeatures(BaseModel):
    tenure_months: int = Field(ge=0)
    monthly_fee: float = Field(gt=0)
    days_active_last_30: int = Field(ge=0, le=30)
    support_tickets_last_30: int = Field(ge=0)
    payment_delay_days: int = Field(ge=0)
    nps_score: int = Field(ge=0, le=10)
    plan: Literal["basic", "standard", "premium"]


class PredictionInput(BaseModel):
    customer_id: str = Field(min_length=1, max_length=128)
    review_text: str = Field(min_length=5, max_length=5000)
    customer: CustomerFeatures


class ServiceMetadata(BaseModel):
    version: str
    git_sha: str


class ModelMetadata(BaseModel):
    name: str
    version: str
    mlflow_run_id: str
    sha256: str
    trained_at: str


class ReviewAnalysis(BaseModel):
    sentiment: Literal["positive", "negative"]
    negative_probability: float


class ChurnPrediction(BaseModel):
    probability: float
    risk: Literal["low", "high"]
    threshold: float


class PredictionOutput(BaseModel):
    request_id: str
    customer_id: str
    service: ServiceMetadata
    model: ModelMetadata
    review_analysis: ReviewAnalysis
    churn_prediction: ChurnPrediction


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    LOGGER.info(
        "request_completed request_id=%s method=%s path=%s status=%s",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
    )
    return response


@lru_cache(maxsize=1)
def load_artifacts():
    bundle = joblib.load(MODEL_PATH)
    metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    return bundle, metadata


def get_artifacts():
    if not MODEL_PATH.exists() or not MODEL_METADATA_PATH.exists():
        raise HTTPException(
            503, "Model is not trained. Run python -m src.churn_ml.training first."
        )
    return load_artifacts()


def public_model_metadata(metadata):
    return {
        "name": metadata["name"],
        "version": metadata["version"],
        "mlflow_run_id": metadata["mlflow_run_id"],
        "sha256": metadata["sha256"],
        "trained_at": metadata["trained_at"],
    }


@app.get("/health")
def health():
    model_ready = MODEL_PATH.exists() and MODEL_METADATA_PATH.exists()
    metadata = None
    if model_ready:
        _, stored_metadata = load_artifacts()
        metadata = public_model_metadata(stored_metadata)
    return {
        "status": "ok",
        "model_ready": model_ready,
        "service": {"version": SERVICE_VERSION, "git_sha": GIT_SHA},
        "model": metadata,
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(payload: PredictionInput, request: Request):
    bundle, metadata = get_artifacts()
    sentiment_model = bundle["sentiment_model"]
    churn_model = bundle["churn_model"]

    sentiment = str(sentiment_model.predict([payload.review_text])[0])
    negative_index = list(sentiment_model.classes_).index("negative")
    negative_probability = float(
        sentiment_model.predict_proba([payload.review_text])[0, negative_index]
    )

    row = payload.customer.model_dump()
    row[SENTIMENT_SCORE_COLUMN] = negative_probability
    churn_probability = float(
        churn_model.predict_proba(pd.DataFrame([row]))[0, 1]
    )
    threshold = float(bundle["threshold"])

    return {
        "request_id": request.state.request_id,
        "customer_id": payload.customer_id,
        "service": {"version": SERVICE_VERSION, "git_sha": GIT_SHA},
        "model": public_model_metadata(metadata),
        "review_analysis": {
            "sentiment": sentiment,
            "negative_probability": round(negative_probability, 4),
        },
        "churn_prediction": {
            "probability": round(churn_probability, 4),
            "risk": "high" if churn_probability >= threshold else "low",
            "threshold": threshold,
        },
    }
