# api/main.py
import numpy as np
import onnxruntime as rt
from fastapi import FastAPI, HTTPException
from schemas import CustomerFeatures, PredictionResponse

app = FastAPI(
    title="Customer Churn Prediction Engine",
    version="1.0.0",
    description="Low-latency customer churn inference using ONNX Runtime."
)

SESSION: rt.InferenceSession = None

@app.on_event("startup")
def load_runtime():
    global SESSION
    model_path = "models/churn_pipeline.onnx"
    try:
        SESSION = rt.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        print("ONNX Runtime session initialized.")
    except Exception as e:
        print(f"Failed to load ONNX model: {e}")

@app.get("/health")
def healthcheck():
    return {"status": "healthy", "engine": "ONNXRuntime-CPU"}

@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    if SESSION is None:
        raise HTTPException(status_code=503, detail="Inference engine not ready.")

    feed_dict = {
        "tenure_months": np.array([[customer.tenure_months]], dtype=np.float32),
        "monthly_charges": np.array([[customer.monthly_charges]], dtype=np.float32),
        "total_charges": np.array([[customer.total_charges]], dtype=np.float32),
        "support_tickets": np.array([[customer.support_tickets]], dtype=np.float32),
        "login_frequency": np.array([[customer.login_frequency]], dtype=np.float32),
        "feature_usage_score": np.array([[customer.feature_usage_score]], dtype=np.float32),
        "contract_type": np.array([[customer.contract_type]], dtype=object),
        "payment_method": np.array([[customer.payment_method]], dtype=object)
    }

    try:
        outputs = SESSION.run(None, feed_dict)
        labels, probabilities = outputs[0], outputs[1]
        churn_prob = float(probabilities[0][1])
        prediction = int(labels[0])

        if churn_prob >= 0.70:
            tier = "HIGH"
        elif churn_prob >= 0.40:
            tier = "MEDIUM"
        else:
            tier = "LOW"

        return PredictionResponse(
            churn_prediction=prediction,
            churn_probability=round(churn_prob, 4),
            risk_tier=tier
        )
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))