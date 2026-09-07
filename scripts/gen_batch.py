import pandas as pd
import numpy as np

np.random.seed(0)
n = 800

df = pd.DataFrame({
    "age": np.random.randint(18, 70, n),
    "income": np.random.randint(20000, 150000, n),
    "credit_history_years": np.random.randint(0, 40, n),
    "loan_amount": np.random.randint(1000, 50000, n),
})
df.to_csv("data/production_batch.csv", index=False)
