FEATURE_COLUMNS = [
    "tenure_months",
    "monthly_fee",
    "days_active_last_30",
    "support_tickets_last_30",
    "payment_delay_days",
    "nps_score",
    "plan",
]

NUMERIC_FEATURES = FEATURE_COLUMNS[:-1]
CATEGORICAL_FEATURES = ["plan"]
