from src.churn_ml.api import CustomerFeatures


def test_api_input_contract():
    customer = CustomerFeatures(
        tenure_months=5, monthly_fee=990, days_active_last_30=14,
        support_tickets_last_30=2, payment_delay_days=1, nps_score=6, plan="standard",
    )
    assert customer.plan == "standard"
