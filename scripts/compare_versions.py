"""Compare two Model Registry versions on the current dataset.

Answers "is the candidate actually better than what's in production right
now?" before promoting it with promote_model.py.

Usage:
    python scripts/compare_versions.py --candidate 3
    python scripts/compare_versions.py --candidate 3 --baseline 2
"""

import argparse
import os
from pathlib import Path

import joblib
import pandas as pd
from mlflow import MlflowClient
from mlflow.artifacts import download_artifacts
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)

from src.churn_ml.training import prepare_churn_features

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "hybrid-review-churn"
DATA_PATH = ROOT / "data/raw/churn_current.csv"


def load_bundle(client, tracking_uri, version=None, alias=None):
    if version is None:
        version = client.get_model_version_by_alias(MODEL_NAME, alias).version
    local_dir = Path(
        download_artifacts(
            artifact_uri=f"models:/{MODEL_NAME}/{version}",
            tracking_uri=tracking_uri,
            dst_path=str(ROOT / ".mlflow_model_cache" / f"compare-v{version}"),
        )
    )
    bundle = joblib.load(local_dir / "hybrid_churn_bundle.joblib")
    return version, bundle


def evaluate(bundle, data):
    sentiment_model = bundle["sentiment_model"]
    churn_model = bundle["churn_model"]

    sentiment_predictions = sentiment_model.predict(data["review_text"])
    churn_probabilities = churn_model.predict_proba(
        prepare_churn_features(data, sentiment_model)
    )[:, 1]

    return {
        "roc_auc": round(roc_auc_score(data["churned"], churn_probabilities), 4),
        "average_precision": round(
            average_precision_score(data["churned"], churn_probabilities), 4
        ),
        "sentiment_accuracy": round(
            accuracy_score(data["review_sentiment"], sentiment_predictions), 4
        ),
        "sentiment_f1": round(
            f1_score(
                data["review_sentiment"],
                sentiment_predictions,
                pos_label="negative",
                zero_division=0,
            ),
            4,
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, help="registry version to evaluate")
    parser.add_argument(
        "--baseline",
        default=None,
        help="registry version to compare against (default: current --baseline-alias)",
    )
    parser.add_argument("--baseline-alias", default="production")
    args = parser.parse_args()

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise SystemExit("MLFLOW_TRACKING_URI is not set")
    if not DATA_PATH.exists():
        raise SystemExit(f"{DATA_PATH} not found. Run `dvc repro` first.")

    client = MlflowClient(tracking_uri=tracking_uri)
    data = pd.read_csv(DATA_PATH)

    baseline_version, baseline_bundle = load_bundle(
        client, tracking_uri, version=args.baseline, alias=args.baseline_alias
    )
    candidate_version, candidate_bundle = load_bundle(
        client, tracking_uri, version=args.candidate
    )

    baseline_metrics = evaluate(baseline_bundle, data)
    candidate_metrics = evaluate(candidate_bundle, data)

    print(f"Evaluated on {DATA_PATH.relative_to(ROOT)} ({len(data)} rows)\n")
    print(f"{'metric':<20}{'baseline v' + str(baseline_version):<18}"
          f"{'candidate v' + str(candidate_version):<18}delta")
    for key in baseline_metrics:
        base, cand = baseline_metrics[key], candidate_metrics[key]
        print(f"{key:<20}{base:<18}{cand:<18}{round(cand - base, 4):+}")


if __name__ == "__main__":
    main()
