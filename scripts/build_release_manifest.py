"""Build a verifiable release passport from repository and ML artifacts."""

import hashlib
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models/hybrid_churn_bundle.joblib"
METADATA_PATH = ROOT / "models/model_metadata.json"
METRICS_PATH = ROOT / "reports/training_metrics.json"
DEFAULT_OUTPUT_PATH = ROOT / "reports/release_manifest.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def current_git_sha() -> str:
    configured = os.getenv("GIT_SHA")
    if configured:
        return configured
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def build_release_manifest(output_path: Path = DEFAULT_OUTPUT_PATH) -> dict:
    required = [
        ROOT / "dvc.lock",
        ROOT / "params.yaml",
        ROOT / "requirements.txt",
        ROOT / "requirements-runtime.txt",
        ROOT / "data/raw/churn_reference.csv",
        ROOT / "data/raw/churn_current.csv",
        MODEL_PATH,
        METADATA_PATH,
        METRICS_PATH,
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing release inputs: " + ", ".join(missing) + ". Run `dvc repro` first."
        )

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    actual_model_sha = sha256(MODEL_PATH)
    if metadata["sha256"] != actual_model_sha:
        raise ValueError("Model digest does not match models/model_metadata.json")

    service_version = os.getenv("SERVICE_VERSION", "0.2.0")
    manifest = {
        "schema_version": "1.0",
        "service": {
            "version": service_version,
            "git_sha": current_git_sha(),
        },
        "pipeline": {
            "dvc_lock_sha256": sha256(ROOT / "dvc.lock"),
            "params_sha256": sha256(ROOT / "params.yaml"),
            "requirements_sha256": sha256(ROOT / "requirements.txt"),
            "runtime_requirements_sha256": sha256(
                ROOT / "requirements-runtime.txt"
            ),
        },
        "data": {
            "reference_sha256": sha256(ROOT / "data/raw/churn_reference.csv"),
            "current_sha256": sha256(ROOT / "data/raw/churn_current.csv"),
        },
        "model": {
            "name": metadata["name"],
            "mlflow_run_id": metadata["mlflow_run_id"],
            "sha256": actual_model_sha,
            "threshold": metadata["threshold"],
        },
        "metrics": metrics,
        "image": {
            "tag": os.getenv("IMAGE_TAG", f"mlops-student:{service_version}"),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    result = build_release_manifest()
    print(json.dumps(result, ensure_ascii=False, indent=2))
