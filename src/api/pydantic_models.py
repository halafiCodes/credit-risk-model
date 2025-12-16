from pydantic import BaseModel
from typing import Optional

class CustomerData(BaseModel):
    CustomerId: int
    Amount: float
    Value: float
    hour: int
    day: int
    month: int
    year: int
    total_amount: float
    avg_amount: float
    transaction_count: int
    std_amount: float
    Amount_log: float
    total_amount_log: float
    avg_amount_log: float
    std_amount_log: float

class PredictionResponse(BaseModel):
    CustomerId: int
    risk_probability: float
