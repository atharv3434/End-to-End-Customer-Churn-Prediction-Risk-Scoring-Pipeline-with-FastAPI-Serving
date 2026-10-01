# src/train.py
import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from lightgbm import LGBMClassifier

from features import build_preprocessor

def train():
    data_path = Path("data/raw/customer_churn.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Run src/generate_data.py first.")
    
    df = pd.read_csv(data_path)
    X = df.drop(columns=["customer_id", "churn"])
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("classifier", LGBMClassifier(
            n_estimators=150,
            learning_rate=0.05,
            num_leaves=31,
            class_weight="balanced",
            random_state=42,
            verbosity=-1
        ))
    ])

    print("Fitting model...")
    pipeline.fit(X_train, y_train)

    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)

    print("\n--- Test Evaluation ---")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print(f"PR-AUC Score:  {pr_auc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    out_model_path = models_dir / "churn_pipeline.joblib"
    joblib.dump(pipeline, out_model_path)
    print(f"Pipeline saved to: {out_model_path}")

if __name__ == "__main__":
    train()