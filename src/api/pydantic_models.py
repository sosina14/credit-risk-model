from pydantic import BaseModel

class CustomerFeatures(BaseModel):
    total_amount: float
    avg_amount: float
    std_amount: float
    min_amount: float
    max_amount: float
    total_value: float
    avg_value: float
    transaction_count: int
    unique_products: int
    unique_channels: int
    fraud_rate: float
    avg_hour: float
    avg_dayofweek: float
    weekend_ratio: float
    Recency: float
    Frequency: float
    Monetary: float

class PredictionResponse(BaseModel):
    customer_id: str
    risk_probability: float
    is_high_risk: int
    risk_label: str