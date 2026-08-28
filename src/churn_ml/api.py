import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Literal
from uuid import uuid4

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from mlflow import MlflowClient
from mlflow.artifacts import download_artifacts
from pydantic import BaseModel, Field

from .features import SENTIMENT_SCORE_COLUMN

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models/hybrid_churn_bundle.joblib"
MODEL_METADATA_PATH = ROOT / "models/model_metadata.json"
SERVICE_VERSION = os.getenv("SERVICE_VERSION", "0.2.0")
GIT_SHA = os.getenv("GIT_SHA", "unknown")
LOGGER = logging.getLogger("uvicorn.error")

MLFLOW_MODEL_NAME = "hybrid-review-churn"
MLFLOW_MODEL_ALIAS = os.getenv("MLFLOW_MODEL_ALIAS", "production")
REGISTRY_CACHE_DIR = ROOT / ".mlflow_model_cache"
# how often a request is allowed to pay the cost of asking the registry
# "is `production` still pointing at the version I have cached?"
REGISTRY_REFRESH_SECONDS = float(os.getenv("MLFLOW_MODEL_REFRESH_SECONDS", "30"))

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


_model_lock = threading.Lock()
_model_cache = {
    "bundle": None,
    "metadata": None,
    "registry_version": None,  # version currently cached in memory
    "checked_at": 0.0,  # monotonic time of the last "is it still current?" check
}


def _download_registry_version(tracking_uri, version_number):
    local_dir = Path(
        download_artifacts(
            artifact_uri=f"models:/{MLFLOW_MODEL_NAME}/{version_number}",
            tracking_uri=tracking_uri,
            dst_path=str(REGISTRY_CACHE_DIR / f"v{version_number}"),
        )
    )
    bundle = joblib.load(local_dir / "hybrid_churn_bundle.joblib")
    metadata = json.loads(
        (local_dir / "model_metadata.json").read_text(encoding="utf-8")
    )
    metadata["registry_version"] = version_number
    metadata["registry_alias"] = MLFLOW_MODEL_ALIAS
    return bundle, metadata


def _load_from_registry():
    """Return the model currently behind MLFLOW_MODEL_ALIAS, reusing the
    in-memory copy unless the alias now points at a different version."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        return None

    now = time.monotonic()
    with _model_lock:
        is_fresh = now - _model_cache["checked_at"] < REGISTRY_REFRESH_SECONDS
        if _model_cache["bundle"] is not None and is_fresh:
            return _model_cache["bundle"], _model_cache["metadata"]

        try:
            client = MlflowClient(tracking_uri=tracking_uri)
            current_version = client.get_model_version_by_alias(
                MLFLOW_MODEL_NAME, MLFLOW_MODEL_ALIAS
            ).version
        except Exception as exc:  # registry unreachable or alias not set yet
            LOGGER.warning("mlflow_registry_unavailable error=%s", exc)
            if _model_cache["bundle"] is not None:
                _model_cache["checked_at"] = now  # avoid hammering a downed server
                return _model_cache["bundle"], _model_cache["metadata"]
            return None

        if current_version == _model_cache["registry_version"]:
            _model_cache["checked_at"] = now
            return _model_cache["bundle"], _model_cache["metadata"]

        LOGGER.info(
            "model_version_changed alias=%s previous=%s current=%s",
            MLFLOW_MODEL_ALIAS,
            _model_cache["registry_version"],
            current_version,
        )
        bundle, metadata = _download_registry_version(tracking_uri, current_version)
        _model_cache.update(
            bundle=bundle,
            metadata=metadata,
            registry_version=current_version,
            checked_at=now,
        )
        return bundle, metadata


def load_artifacts():
    from_registry = _load_from_registry()
    if from_registry is not None:
        return from_registry
    if not MODEL_PATH.exists() or not MODEL_METADATA_PATH.exists():
        raise FileNotFoundError("model artifacts not found in registry or locally")
    bundle = joblib.load(MODEL_PATH)
    metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    return bundle, metadata


def get_artifacts():
    try:
        return load_artifacts()
    except FileNotFoundError as exc:
        raise HTTPException(
            503, "Model is not trained. Run python -m src.churn_ml.training first."
        ) from exc


@app.post("/admin/reload-model")
def reload_model():
    """Force an immediate alias check instead of waiting for the next
    scheduled one (up to MLFLOW_MODEL_REFRESH_SECONDS away)."""
    with _model_lock:
        _model_cache["checked_at"] = 0.0
    return {"status": "will_check_registry_on_next_request"}


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
    try:
        _, stored_metadata = load_artifacts()
    except FileNotFoundError:
        return {
            "status": "ok",
            "model_ready": False,
            "service": {"version": SERVICE_VERSION, "git_sha": GIT_SHA},
            "model": None,
        }
    return {
        "status": "ok",
        "model_ready": True,
        "service": {"version": SERVICE_VERSION, "git_sha": GIT_SHA},
        "model": public_model_metadata(stored_metadata),
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
