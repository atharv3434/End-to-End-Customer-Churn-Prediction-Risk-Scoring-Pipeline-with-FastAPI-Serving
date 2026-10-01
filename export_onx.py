# src/export_onnx.py
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, StringTensorType

def export_pipeline_to_onnx():
    model_path = Path("models/churn_pipeline.joblib")
    pipeline = joblib.load(model_path)

    # Define schema specifications for ONNX inputs
    initial_type = [
        ("tenure_months", FloatTensorType([None, 1])),
        ("monthly_charges", FloatTensorType([None, 1])),
        ("total_charges", FloatTensorType([None, 1])),
        ("support_tickets", FloatTensorType([None, 1])),
        ("login_frequency", FloatTensorType([None, 1])),
        ("feature_usage_score", FloatTensorType([None, 1])),
        ("contract_type", StringTensorType([None, 1])),
        ("payment_method", StringTensorType([None, 1]))
    ]

    onnx_model = convert_sklearn(
        pipeline,
        initial_types=initial_type,
        target_opset=14,
        options={type(pipeline.named_steps["classifier"]): {"zipmap": False}}
    )

    out_onnx_path = Path("models/churn_pipeline.onnx")
    with open(out_onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print(f"ONNX model exported to: {out_onnx_path}")

if __name__ == "__main__":
    export_pipeline_to_onnx()