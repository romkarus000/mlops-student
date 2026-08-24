from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .features import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models/churn_model.joblib"
app = FastAPI(title="Churn prediction API", version="0.1.0")


class CustomerFeatures(BaseModel):
    tenure_months: int = Field(ge=0)
    monthly_fee: float = Field(gt=0)
    days_active_last_30: int = Field(ge=0, le=30)
    support_tickets_last_30: int = Field(ge=0)
    payment_delay_days: int = Field(ge=0)
    nps_score: int = Field(ge=0, le=10)
    plan: str


def get_model():
    if not MODEL_PATH.exists():
        raise HTTPException(503, "Model is not trained. Run python -m src.churn_ml.training first.")
    return joblib.load(MODEL_PATH)


@app.get("/health")
def health():
    return {"status": "ok", "model_ready": MODEL_PATH.exists()}


@app.post("/predict")
def predict(customer: CustomerFeatures):
    model = get_model()
    row = pd.DataFrame([customer.model_dump()])[FEATURE_COLUMNS]
    probability = float(model.predict_proba(row)[0, 1])
    return {"churn_probability": round(probability, 4), "risk": "high" if probability >= 0.5 else "low"}
