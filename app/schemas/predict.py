from pydantic import BaseModel
from typing import Optional

class Coordinates(BaseModel):
    lat: float
    lng: float

class PredictOut(BaseModel):
    type: str
    color: str
    confidence: float
    description: str
    coordinates: Optional[Coordinates] = None
