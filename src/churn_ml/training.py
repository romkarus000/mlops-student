import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import (
    CHURN_FEATURE_COLUMNS,
    CHURN_NUMERIC_FEATURES,
    CUSTOMER_CATEGORICAL_FEATURES,
    CUSTOMER_FEATURE_COLUMNS,
    SENTIMENT_SCORE_COLUMN,
)

ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "hybrid-review-churn"


def load_params():
    with open(ROOT / "params.yaml", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_sentiment_pipeline(random_state=42):
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
            (
                "model",
                LogisticRegression(
                    max_iter=1000, solver="liblinear", random_state=random_state
                ),
            ),
        ]
    )


def build_churn_pipeline(random_state=42):
    preprocess = ColumnTransformer(
        [
            ("numeric", StandardScaler(), CHURN_NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CUSTOMER_CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline(
        [
            ("preprocess", preprocess),
            (
                "model",
                LogisticRegression(
                    max_iter=1000, solver="liblinear", random_state=random_state
                ),
            ),
        ]
    )


def negative_sentiment_probability(sentiment_model, reviews):
    negative_index = list(sentiment_model.classes_).index("negative")
    return sentiment_model.predict_proba(reviews)[:, negative_index]


def prepare_churn_features(frame, sentiment_model):
    features = frame[CUSTOMER_FEATURE_COLUMNS].copy()
    features[SENTIMENT_SCORE_COLUMN] = negative_sentiment_probability(
        sentiment_model, frame["review_text"]
    )
    return features[CHURN_FEATURE_COLUMNS]


def train():
    params = load_params()["training"]
    data = pd.read_csv(ROOT / "data/raw/churn_reference.csv")
    train_data, test_data = train_test_split(
        data,
        test_size=params["test_size"],
        random_state=params["random_state"],
        stratify=data["churned"],
    )

    sentiment_model = build_sentiment_pipeline(params["random_state"])
    sentiment_model.fit(train_data["review_text"], train_data["review_sentiment"])
    sentiment_predictions = sentiment_model.predict(test_data["review_text"])

    churn_model = build_churn_pipeline(params["random_state"])
    churn_model.fit(
        prepare_churn_features(train_data, sentiment_model), train_data["churned"]
    )
    churn_probabilities = churn_model.predict_proba(
        prepare_churn_features(test_data, sentiment_model)
    )[:, 1]

    metrics = {
        "roc_auc": round(roc_auc_score(test_data["churned"], churn_probabilities), 4),
        "average_precision": round(
            average_precision_score(test_data["churned"], churn_probabilities), 4
        ),
        "sentiment_accuracy": round(
            accuracy_score(test_data["review_sentiment"], sentiment_predictions), 4
        ),
        "sentiment_f1": round(
            f1_score(
                test_data["review_sentiment"],
                sentiment_predictions,
                pos_label="negative",
                zero_division=0,
            ),
            4,
        ),
        "train_rows": len(train_data),
        "test_rows": len(test_data),
    }

    models_dir = ROOT / "models"
    reports_dir = ROOT / "reports"
    models_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)
    model_path = models_dir / "hybrid_churn_bundle.joblib"
    metadata_path = models_dir / "model_metadata.json"
    report_path = reports_dir / "training_metrics.json"

    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", f"file://{ROOT / 'mlruns'}")
    )
    mlflow.set_experiment("hybrid-review-churn-demo")
    with mlflow.start_run(run_name="hybrid-sentiment-churn") as run:
        model_version = run.info.run_id
        bundle = {
            "schema_version": "1.0",
            "sentiment_model": sentiment_model,
            "churn_model": churn_model,
            "customer_features": CUSTOMER_FEATURE_COLUMNS,
            "sentiment_score_column": SENTIMENT_SCORE_COLUMN,
            "threshold": params["threshold"],
        }
        joblib.dump(bundle, model_path)
        model_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
        metadata = {
            "name": MODEL_NAME,
            "version": model_version,
            "mlflow_run_id": model_version,
            "sha256": model_sha256,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "threshold": params["threshold"],
            "metrics": metrics,
        }
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        mlflow.log_params(params)
        mlflow.log_param("sentiment_model", "tfidf_logistic_regression")
        mlflow.log_param("churn_model", "logistic_regression")
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(model_path), artifact_path="model")
        mlflow.log_artifact(str(metadata_path), artifact_path="model")
        mlflow.log_artifact(str(report_path), artifact_path="reports")

        registered_model = mlflow.register_model(
            model_uri=f"runs:/{run.info.run_id}/model", name=MODEL_NAME
        )

    return {
        **metrics,
        "model_version": model_version,
        "model_sha256": model_sha256,
        "registry_version": registered_model.version,
    }


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
