from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def make_dataset(rows, seed, drift=False):
    rng = np.random.default_rng(seed)
    plans = rng.choice(["basic", "standard", "premium"], rows, p=[0.48, 0.37, 0.15] if not drift else [0.30, 0.34, 0.36])
    active_days = np.clip(rng.normal(18 if not drift else 12, 6, rows).round(), 0, 30).astype(int)
    tickets = rng.poisson(1.4 if not drift else 2.8, rows)
    delay = rng.poisson(1.7 if not drift else 3.3, rows)
    nps = np.clip(rng.normal(7 if not drift else 5.5, 2, rows).round(), 0, 10).astype(int)
    fee = np.select([plans == "basic", plans == "standard", plans == "premium"], [490, 990, 1990]).astype(float)
    fee += rng.normal(0, 35, rows)
    tenure = rng.integers(1, 37, rows)
    logit = -0.6 - 0.08 * active_days + 0.33 * tickets + 0.22 * delay - 0.18 * nps - 0.025 * tenure
    probability = 1 / (1 + np.exp(-logit))
    churned = rng.binomial(1, probability)
    return pd.DataFrame({
        "customer_id": [f"C{seed}{i:05d}" for i in range(rows)],
        "tenure_months": tenure,
        "monthly_fee": fee.round(2),
        "days_active_last_30": active_days,
        "support_tickets_last_30": tickets,
        "payment_delay_days": delay,
        "nps_score": nps,
        "plan": plans,
        "churned": churned,
    })


def main():
    params = yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))["data"]
    destination = ROOT / "data/raw"
    destination.mkdir(parents=True, exist_ok=True)
    make_dataset(params["reference_rows"], params["random_state"]).to_csv(destination / "churn_reference.csv", index=False)
    make_dataset(params["current_rows"], params["random_state"] + 1, drift=True).to_csv(destination / "churn_current.csv", index=False)


if __name__ == "__main__":
    main()
