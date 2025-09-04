# Terrain Backend (FastAPI)

A minimal backend to serve terrain classification using your own `.pkl` model.

## Quick start

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
# source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

# copy your model to models/terrain_model.pkl
# (optional) edit models/labels.json

copy .env.example .env   # or cp .env.example .env
# edit .env if needed

uvicorn app.main:app --reload --port 8000
# Open http://localhost:8000/docs
```

### API
- `POST /api/predict` (multipart form, field: `file`): returns `{ type, color, confidence, description, coordinates }`
- `GET /health`

### Notes
- If your model is scikit-learn, you can remove `torch` from `requirements.txt`.
- Adjust `IMG_SIZE` in `.env` or change `preprocess()` in `app/services/model.py` for custom pipelines.
