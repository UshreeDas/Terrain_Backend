from fastapi import APIRouter, File, UploadFile, HTTPException
from app.schemas.predict import PredictOut
from app.utils.image import load_pil_image
from app.services.model import predict_probs, CLASS_NAMES, top1
import numpy as np

router = APIRouter(tags=["predict"])

DESCRIPTIONS = {
    "mountain": "Elevated rugged terrain; steep slopes and high relief.",
    "forest":   "Dense vegetation canopy; tree cover dominates the surface.",
    "desert":   "Sparse vegetation; sand/rock with arid patterns.",
    "coastal":  "Shoreline features; water–land boundary and beach forms.",
    "plain":    "Low relief, broad flat areas; grassland/cropland patterns.",
}

def _color_key(name: str) -> str:
    return name

@router.post("/predict", response_model=PredictOut)
async def predict(file: UploadFile = File(...)):
    try:
        raw = await file.read()
        img = load_pil_image(raw)
        probs = predict_probs(img)
        if probs.size != len(CLASS_NAMES):
            n = len(CLASS_NAMES)
            if probs.size < n:
                probs = np.pad(probs, (0, n - probs.size))
            else:
                probs = probs[:n]
        idx, conf = top1(probs)
        label = CLASS_NAMES[idx] if 0 <= idx < len(CLASS_NAMES) else "plain"
        return PredictOut(
            type=label,
            color=_color_key(label),
            confidence=round(float(conf) * 100, 2),
            description=DESCRIPTIONS.get(label, "Terrain classification result."),
            coordinates=None,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")
