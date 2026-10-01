# api/schemas.py
from typing import List
from pydantic import BaseModel, Field

class CustomerFeatures(BaseModel):
    tenure_months: float = Field(..., ge=0, example=12.0)
    monthly_charges: float = Field(..., ge=0, example=75.50)
    total_charges: float = Field(..., ge=0, example=906.0)
    support_tickets: float = Field(..., ge=0, example=2.0)
    login_frequency: float = Field(..., ge=0, example=8.0)
    feature_usage_score: float = Field(..., ge=0, le=100, example=45.2)
    contract_type: str = Field(..., example="Month-to-Month")
    payment_method: str = Field(..., example="Credit-Card")

class PredictionResponse(BaseModel):
    churn_prediction: int
    churn_probability: float
    risk_tier: str