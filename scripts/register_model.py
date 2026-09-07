import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()

experiment = client.get_experiment_by_name("credit-scoring")
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["start_time DESC"],
    max_results=1,
)
run_id = runs[0].info.run_id

result = mlflow.register_model(f"runs:/{run_id}/model", "credit-model")
client.set_registered_model_alias("credit-model", "production", result.version)
print(f"Registered credit-model version {result.version} as @production")
