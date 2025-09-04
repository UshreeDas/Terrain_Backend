from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from app.routers.predict import router as predict_router

def _origins():
    raw = os.getenv("CORS_ORIGINS", "http://localhost:8080")
    return [s.strip() for s in raw.split(",") if s.strip()]

app = FastAPI(title="Terrain Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router, prefix="/api")

@app.get("/health")
def health():
    return {"ok": True}
