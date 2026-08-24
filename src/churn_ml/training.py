import json
import os
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES

ROOT = Path(__file__).resolve().parents[2]


def load_params():
    with open(ROOT / "params.yaml", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_pipeline():
    preprocess = ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline([("preprocess", preprocess), ("model", LogisticRegression(max_iter=1000))])


def train():
    params = load_params()["training"]
    data = pd.read_csv(ROOT / "data/raw/churn_reference.csv")
    X_train, X_test, y_train, y_test = train_test_split(
        data[FEATURE_COLUMNS], data["churned"], test_size=params["test_size"],
        random_state=params["random_state"], stratify=data["churned"],
    )
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    metrics = {
        "roc_auc": round(roc_auc_score(y_test, probabilities), 4),
        "average_precision": round(average_precision_score(y_test, probabilities), 4),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    }
    Path(ROOT / "models").mkdir(exist_ok=True)
    Path(ROOT / "reports").mkdir(exist_ok=True)
    model_path = ROOT / "models/churn_model.joblib"
    report_path = ROOT / "reports/training_metrics.json"
    joblib.dump(pipeline, model_path)
    report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # MLflow — опциональная инфраструктурная зависимость: без неё модель всё равно
    # обучается, а на вебинаре можно отдельно показать, что даёт tracking server.
    try:
        import mlflow
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", f"file://{ROOT / 'mlruns'}"))
        mlflow.set_experiment("churn-demo")
        with mlflow.start_run(run_name="baseline-logistic-regression"):
            mlflow.log_params(params)
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, float)})
            mlflow.log_artifact(str(model_path), artifact_path="model")
            mlflow.log_artifact(str(report_path), artifact_path="reports")
    except ModuleNotFoundError:
        print("MLflow is not installed: model and metrics were saved locally.")
    return metrics


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
