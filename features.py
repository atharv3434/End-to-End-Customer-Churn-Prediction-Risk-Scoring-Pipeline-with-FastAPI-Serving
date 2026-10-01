# src/features.py
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERICAL_FEATURES = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "support_tickets",
    "login_frequency",
    "feature_usage_score"
]

CATEGORICAL_FEATURES = [
    "contract_type",
    "payment_method"
]

def build_preprocessor() -> ColumnTransformer:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ],
        remainder="drop"
    )
    return preprocessor