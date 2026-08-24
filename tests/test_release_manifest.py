import hashlib
import json
from pathlib import Path

from scripts.build_release_manifest import build_release_manifest


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_release_manifest_links_code_data_pipeline_and_model(tmp_path):
    output_path = tmp_path / "release_manifest.json"
    manifest = build_release_manifest(output_path)

    assert output_path.exists()
    assert json.loads(output_path.read_text(encoding="utf-8")) == manifest
    assert manifest["service"]["git_sha"]
    assert manifest["pipeline"]["dvc_lock_sha256"] == digest(ROOT / "dvc.lock")
    assert manifest["pipeline"]["runtime_requirements_sha256"] == digest(
        ROOT / "requirements-runtime.txt"
    )
    assert manifest["data"]["reference_sha256"] == digest(
        ROOT / "data/raw/churn_reference.csv"
    )
    assert manifest["model"]["sha256"] == digest(
        ROOT / "models/hybrid_churn_bundle.joblib"
    )
    assert manifest["model"]["mlflow_run_id"]
    assert manifest["image"]["tag"].startswith("mlops-student:")
