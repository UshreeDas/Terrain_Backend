import os, json, numpy as np
from PIL import Image
from typing import List
import joblib

MODEL_PATH = os.getenv("MODEL_PATH", "models/terrain_model.pkl")
LABELS_PATH = os.getenv("LABELS_PATH", "models/labels.json")
IMG_SIZE = int(os.getenv("IMG_SIZE", "224"))

# Try importing torch if available
_TORCH = None
try:
    import torch
    _TORCH = torch
except Exception:
    _TORCH = None

_FALLBACK_LABELS = ["mountain", "forest", "desert", "coastal", "plain"]

def _load_labels() -> List[str]:
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return _FALLBACK_LABELS

CLASS_NAMES = _load_labels()

_model = None
_is_torch = False

def load_model():
    global _model, _is_torch
    if _model is not None:
        return _model
    # Try joblib (sklearn/xgb or pickled wrapper)
    try:
        _model = joblib.load(MODEL_PATH)
        _is_torch = False
        return _model
    except Exception:
        pass
    # Try torch if available
    if _TORCH:
        try:
            _model = _TORCH.load(MODEL_PATH, map_location="cpu")
            _model.eval()
            _is_torch = True
            return _model
        except Exception:
            pass
    raise RuntimeError(f"Could not load model from '{MODEL_PATH}'. Ensure it's a joblib/sklearn pickle or a torch model.")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess(img: Image.Image) -> np.ndarray:
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img).astype("float32") / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr

def predict_probs(img: Image.Image) -> np.ndarray:
    model = load_model()
    x = preprocess(img)
    if _is_torch:
        import torch
        x_t = torch.tensor(x).permute(2,0,1).unsqueeze(0)  # 1x3xHxW
        with torch.no_grad():
            logits = model(x_t)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        return probs
    else:
        x_flat = x.reshape(1, -1)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(x_flat)[0]
            return np.array(probs, dtype=np.float32)
        pred = model.predict(x_flat)
        probs = np.zeros(len(CLASS_NAMES), dtype=np.float32)
        try:
            idx = int(pred[0])
        except Exception:
            idx = 0
        probs[idx] = 1.0
        return probs

def top1(probs: np.ndarray):
    idx = int(np.argmax(probs))
    return idx, float(probs[idx])
