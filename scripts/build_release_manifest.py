"""Build a verifiable release passport (release manifest) for ml-service.

Links in one JSON file: the git commit, the DVC pipeline state (dvc.lock,
params, data), the trained model (file digest + MLflow Registry version),
the metrics and the Docker image tag.

Usage:
    python scripts/build_release_manifest.py [--output PATH] [--skip-registry]

Environment:
    SERVICE_VERSION      service version, default "0.1.0"
    IMAGE_TAG            Docker image tag, default "ml-service:<git sha>"
    GIT_SHA              commit sha, default `git rev-parse HEAD`
    MLFLOW_TRACKING_URI  default "http://localhost:5000"
"""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "reports" / "release_manifest.json"

MODEL_NAME = "credit-model"
MODEL_ALIAS = "production"

REQUIRED_INPUTS = [
    "dvc.lock",
    "params.yaml",
    "requirements.txt",
    "data/train.csv",
    "models/model.pkl",
    "metrics.json",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_sha() -> str:
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


def registry_entry() -> dict:
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    version = MlflowClient().get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    return {
        "name": MODEL_NAME,
        "alias": MODEL_ALIAS,
        "version": version.version,
        "mlflow_run_id": version.run_id,
    }


def build_manifest(output: Path, skip_registry: bool) -> dict:
    missing = [name for name in REQUIRED_INPUTS if not (ROOT / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing release inputs: "
            + ", ".join(missing)
            + ". Run `dvc repro` (and `dvc pull` / scripts/gen_train_data.py for data) first."
        )

    commit = git_sha()
    service_version = os.getenv("SERVICE_VERSION", "0.1.0")

    model = {"file": "models/model.pkl", "sha256": sha256(ROOT / "models/model.pkl")}
    if skip_registry:
        model["registry"] = None
    else:
        model["registry"] = registry_entry()

    manifest = {
        "schema_version": "1.0",
        "service": {"version": service_version, "git_sha": commit},
        "pipeline": {
            "dvc_lock_sha256": sha256(ROOT / "dvc.lock"),
            "params_sha256": sha256(ROOT / "params.yaml"),
            "requirements_sha256": sha256(ROOT / "requirements.txt"),
        },
        "data": {"train_sha256": sha256(ROOT / "data/train.csv")},
        "model": model,
        "metrics": json.loads((ROOT / "metrics.json").read_text(encoding="utf-8")),
        "image": {"tag": os.getenv("IMAGE_TAG", f"ml-service:{commit}")},
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--skip-registry",
        action="store_true",
        help="do not query the MLflow Model Registry (no server available)",
    )
    args = parser.parse_args()

    manifest = build_manifest(args.output, args.skip_registry)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
