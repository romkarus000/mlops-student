import pandas as pd
import numpy as np

np.random.seed(42)
n = 800

age = np.random.randint(18, 70, n)
income = np.random.randint(20000, 150000, n)
credit_history_years = np.random.randint(0, 40, n)
loan_amount = np.random.randint(1000, 50000, n)

score = (
    0.02 * income / 1000
    + 0.3 * credit_history_years
    - 0.0004 * loan_amount
    - 0.1 * (age < 25)
    + np.random.normal(0, 5, n)
)
target = (score > np.median(score)).astype(int)

df = pd.DataFrame({
    "age": age, "income": income,
    "credit_history_years": credit_history_years,
    "loan_amount": loan_amount, "target": target,
})
df.to_csv("data/train.csv", index=False)
