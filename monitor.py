import pandas as pd

from evidently import Report, Dataset, DataDefinition
from evidently.presets import DataDriftPreset

reference_df = pd.read_csv("data/train.csv").drop(columns=["target"])
current_df = pd.read_csv("data/production_batch.csv")

reference = Dataset.from_pandas(reference_df, data_definition=DataDefinition())
current = Dataset.from_pandas(current_df, data_definition=DataDefinition())

report = Report([DataDriftPreset()])
run = report.run(current, reference)
run.save_html("drift_report.html")

print(run.dict())

from evidently import Report
from evidently.presets import DataDriftPreset

gate = Report([DataDriftPreset()], include_tests=True)
gate_run = gate.run(current, reference)
gate_dict = gate_run.dict()

drifted_share = gate_dict["metrics"][0]["value"]["share"]
THRESHOLD = 0.2
if drifted_share > THRESHOLD:
    print(f"DRIFT GATE FAILED: {drifted_share:.0%} columns drifted (threshold {THRESHOLD:.0%})")
    exit(1)
else:
    print(f"DRIFT GATE OK: {drifted_share:.0%} columns drifted")
