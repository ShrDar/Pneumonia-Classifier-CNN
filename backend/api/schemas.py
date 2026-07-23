from pydantic import BaseModel


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probability: float
    gradcam_image: str | None = None
