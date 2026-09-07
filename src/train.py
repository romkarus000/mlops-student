import json
import pickle

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

params = yaml.safe_load(open("params.yaml"))["train"]

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("credit-scoring")

df = pd.read_csv("data/train.csv")
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

with mlflow.start_run():
    model = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        random_state=42,
    )
    model.fit(X_train, y_train)

    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print(f"roc_auc: {auc:.4f}")

    mlflow.log_param("n_estimators", params["n_estimators"])
    mlflow.log_param("max_depth", params["max_depth"])
    mlflow.log_metric("roc_auc", auc)
    mlflow.sklearn.log_model(model, name="model")

with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("metrics.json", "w") as f:
    json.dump({"roc_auc": auc}, f, indent=2)
