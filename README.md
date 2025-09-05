# Terrain Recognition API (Flask)

This service operationalizes the notebook logic from **Terrain_recognition.ipynb** as a HTTP API.

## What it does
- Loads two data artifacts produced by the notebook:
  - `model/coordinates_data.pkl` — state bounding boxes (columns: `lat_min`, `lat_max`, `lon_min`, `lon_max`, `name`)
  - `model/geological_data.pkl`  — geology per state (columns include: `State`, `Geological_Dominance`, `Rock_Type`, `Soil_Type`, `Earthquake_Zone`, `Average_Elevation_m`, ...)
- Given a latitude/longitude, it looks up the state via bounding boxes (rule-based, not ML) and returns the state's geological info.
- Optionally renders a synthetic 3D terrain image (PNG) using the state's average elevation.

## Quickstart

```bash
cd terrain_api
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run
python app.py
# Service listens on http://localhost:8000
```

### Health check
```bash
curl -s http://localhost:8000/health | jq
```

### Predict (JSON in/out)
```bash
curl -s -X POST http://localhost:8000/predict   -H "Content-Type: application/json"   -d '{"latitude": 19.07, "longitude": 72.87, "include_terrain_png": true }' | jq -r "."
```

### Terrain image (direct PNG)
```bash
curl -L "http://localhost:8000/terrain.png?lat=19.07&lon=72.87&grid_size=60&variation=60" -o terrain.png
```

## Notes
- The logic mirrors the notebook: a **bounding-box** lookup (not polygon GIS), and geology fetch by `State` exact match.
- For portability, the app attempts a pickle shim for environments where `numpy._core` changed. If you re-generate pickles, do so with your current pandas/numpy versions to avoid compatibility issues.
- You can set custom paths via env vars:
  - `COORDS_PKL=model/coordinates_data.pkl`
  - `GEO_PKL=model/geological_data.pkl`


## CORS
By default, CORS is **open** (`*`). To restrict origins, set:
```bash
export CORS_ORIGINS="https://yourdomain.com,https://staging.yourdomain.com"
```
The server will then allow credentials for those origins.
