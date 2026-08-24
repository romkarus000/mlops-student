from fastapi.testclient import TestClient

from src.churn_ml.api import CustomerFeatures, PredictionInput, app


PREDICTION_PAYLOAD = {
    "customer_id": "C-TEST-001",
    "review_text": "Поддержка не отвечает, хочу отменить подписку",
    "customer": {
        "tenure_months": 5,
        "monthly_fee": 990,
        "days_active_last_30": 14,
        "support_tickets_last_30": 2,
        "payment_delay_days": 1,
        "nps_score": 6,
        "plan": "standard",
    },
}


def test_api_input_contract():
    payload = PredictionInput(**PREDICTION_PAYLOAD)
    assert isinstance(payload.customer, CustomerFeatures)
    assert payload.review_text.startswith("Поддержка")


def test_predict_returns_traceability_and_hybrid_prediction():
    client = TestClient(app)
    response = client.post(
        "/predict",
        json=PREDICTION_PAYLOAD,
        headers={"X-Request-ID": "test-request-001"},
    )
    assert response.status_code == 200
    body = response.json()
    assert response.headers["X-Request-ID"] == "test-request-001"
    assert body["request_id"] == "test-request-001"
    assert body["customer_id"] == "C-TEST-001"
    assert body["service"]["version"]
    assert body["model"]["version"] == body["model"]["mlflow_run_id"]
    assert body["review_analysis"]["sentiment"] == "negative"
    assert 0 <= body["review_analysis"]["negative_probability"] <= 1
    assert 0 <= body["churn_prediction"]["probability"] <= 1
    assert body["churn_prediction"]["threshold"] == 0.5


def test_health_exposes_service_and_model_versions():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["model_ready"] is True
    assert body["service"]["version"]
    assert body["model"]["sha256"]


def test_negative_review_increases_risk_for_same_customer():
    client = TestClient(app)
    negative = client.post("/predict", json=PREDICTION_PAYLOAD).json()
    positive_payload = {
        **PREDICTION_PAYLOAD,
        "review_text": "Сервис работает отлично, всё удобно и понятно",
    }
    positive = client.post("/predict", json=positive_payload).json()

    assert negative["review_analysis"]["negative_probability"] > positive[
        "review_analysis"
    ]["negative_probability"]
    assert negative["churn_prediction"]["probability"] > positive[
        "churn_prediction"
    ]["probability"]
