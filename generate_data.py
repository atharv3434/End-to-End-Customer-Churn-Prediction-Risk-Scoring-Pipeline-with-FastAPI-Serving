# src/generate_data.py
import numpy as np
import pandas as pd
from pathlib import Path

def generate_customer_churn_dataset(n_samples: int = 10000, random_state: int = 42) -> pd.DataFrame:
    np.random.seed(random_state)
    
    customer_ids = [f"CUST-{10000 + i}" for i in range(n_samples)]
    tenure_months = np.random.exponential(scale=18, size=n_samples).clip(1, 72).astype(int)
    contract_type = np.random.choice(["Month-to-Month", "One-Year", "Two-Year"], size=n_samples, p=[0.55, 0.25, 0.20])
    payment_method = np.random.choice(["Credit-Card", "Bank-Transfer", "Electronic-Check"], size=n_samples, p=[0.4, 0.3, 0.3])
    monthly_charges = np.random.normal(loc=65, scale=25, size=n_samples).clip(18.0, 120.0).round(2)
    total_charges = (monthly_charges * tenure_months + np.random.normal(0, 10, size=n_samples)).clip(lower=18.0).round(2)
    
    support_tickets = np.random.poisson(lam=1.5, size=n_samples)
    login_frequency_per_week = np.random.negative_binomial(n=5, p=0.4, size=n_samples)
    feature_usage_score = np.random.beta(a=2, b=5, size=n_samples) * 100
    
    # Non-linear probability of churn
    logit = (
        -1.5
        + (contract_type == "Month-to-Month") * 1.2
        - (contract_type == "Two-Year") * 1.0
        + (support_tickets > 3) * 1.4
        - (tenure_months / 12.0) * 0.4
        + (monthly_charges / 100.0) * 0.8
        - (login_frequency_per_week / 10.0) * 0.9
        - (feature_usage_score / 100.0) * 1.1
    )
    churn_prob = 1.0 / (1.0 + np.exp(-logit))
    churn = (np.random.rand(n_samples) < churn_prob).astype(int)

    df = pd.DataFrame({
        "customer_id": customer_ids,
        "tenure_months": tenure_months,
        "contract_type": contract_type,
        "payment_method": payment_method,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "support_tickets": support_tickets,
        "login_frequency": login_frequency_per_week,
        "feature_usage_score": feature_usage_score.round(2),
        "churn": churn
    })
    return df

if __name__ == "__main__":
    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    df = generate_customer_churn_dataset(n_samples=12000)
    out_path = out_dir / "customer_churn.csv"
    df.to_csv(out_path, index=False)
    print(f"Data saved to {out_path} | Churn Rate: {df['churn'].mean():.2%}")