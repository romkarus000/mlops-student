from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import mlflow
import mlflow.pyfunc
import pandas as pd

app = FastAPI(title="ml-service")

mlflow.set_tracking_uri("http://localhost:5000")

ALLOWED_ALIASES = {"production", "staging"}
FEATURE_COLUMNS = ["age", "income", "credit_history_years", "loan_amount"]

_model_cache: dict[str, "mlflow.pyfunc.PyFuncModel"] = {}


def get_model(alias: str):
    if alias not in ALLOWED_ALIASES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown alias '{alias}'. Allowed: {sorted(ALLOWED_ALIASES)}",
        )
    if alias not in _model_cache:
        try:
            _model_cache[alias] = mlflow.pyfunc.load_model(f"models:/credit-model@{alias}")
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Model alias '{alias}' unavailable: {exc}",
            )
    return _model_cache[alias]


class PredictRequest(BaseModel):
    features: list[float]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest, alias: str = Query("production")):
    if len(req.features) != len(FEATURE_COLUMNS):
        raise HTTPException(
            status_code=422,
            detail=f"Expected {len(FEATURE_COLUMNS)} features: {FEATURE_COLUMNS}",
        )
    model = get_model(alias)
    df = pd.DataFrame([req.features], columns=FEATURE_COLUMNS)
    prediction = model.predict(df)
    return {"prediction": float(prediction[0]), "model_alias": alias}
