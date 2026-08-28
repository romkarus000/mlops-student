"""Promote a Model Registry version to an alias (e.g. production).

Run after comparing versions in the MLflow UI or via `mlflow models list`:

    python scripts/promote_model.py --version 3
    python scripts/promote_model.py --version 2   # rollback
"""

import argparse
import os

from mlflow import MlflowClient

MODEL_NAME = "hybrid-review-churn"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, help="Registry version to promote")
    parser.add_argument("--alias", default="production")
    parser.add_argument("--model-name", default=MODEL_NAME)
    args = parser.parse_args()

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise SystemExit("MLFLOW_TRACKING_URI is not set")

    client = MlflowClient(tracking_uri=tracking_uri)
    client.set_registered_model_alias(args.model_name, args.alias, args.version)
    print(f"{args.model_name} v{args.version} -> alias '{args.alias}'")


if __name__ == "__main__":
    main()
