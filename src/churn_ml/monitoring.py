from pathlib import Path

import json
import pandas as pd

from .features import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[2]


def monitor():
    reference = pd.read_csv(ROOT / "data/raw/churn_reference.csv")[FEATURE_COLUMNS]
    current = pd.read_csv(ROOT / "data/raw/churn_current.csv")[FEATURE_COLUMNS]
    Path(ROOT / "reports").mkdir(exist_ok=True)
    output = ROOT / "reports/monitoring_report.html"
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference, current_data=current).save_html(output)
    except ModuleNotFoundError:
        # Лёгкий fallback сохраняет понятный отчёт даже до установки Evidently.
        shifts = {}
        for column in FEATURE_COLUMNS:
            if pd.api.types.is_numeric_dtype(reference[column]):
                shifts[column] = round(float(current[column].mean() - reference[column].mean()), 3)
            else:
                shifts[column] = "category_distribution_changed"
        output.write_text(
            "<h1>Monitoring report (fallback)</h1>"
            "<p>Evidently is not installed. Mean shifts / categorical changes:</p>"
            f"<pre>{json.dumps(shifts, indent=2)}</pre>", encoding="utf-8"
        )
    return output


if __name__ == "__main__":
    print(f"Monitoring report saved to: {monitor()}")
