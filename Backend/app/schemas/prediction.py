from pydantic import BaseModel, Field
from typing import Optional

class PredictionResponse(BaseModel):
    label: str = Field(..., description="Predicted class label or 'no_leaf'")
    confidence: float = Field(..., description="Confidence score between 0.0 and 1.0")
    message: Optional[str] = Field(None, description="Optional informational message")

class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message description")
