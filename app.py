# app.py
from __future__ import annotations

import io
import os
import sys
import time
import base64
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_file, abort
from flask_cors import CORS

# ---------------------------------------------------------------------
# Robust unpickling shim (handles environments where "numpy._core" path changed)
# ---------------------------------------------------------------------
try:
    import numpy.core as _np_core  # noqa: F401
    sys.modules.setdefault("numpy._core", _np_core)
except Exception:
    pass

APP_VERSION = "1.0.0"

app = Flask(__name__)

# Enable CORS for all routes and origins.
CORS(app, resources={r"/*": {"origins": "*"}})

COORDS_PKL = os.getenv("COORDS_PKL", "model/coordinates_data.pkl")
GEO_PKL    = os.getenv("GEO_PKL",    "model/geological_data.pkl")

_loaded = {
    "coords_df": None,
    "geo_df": None,
    "loaded_at": None,
}


def _safe_load_pickle(path: str) -> Any:
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    coords_df = _safe_load_pickle(COORDS_PKL)
    geo_df = _safe_load_pickle(GEO_PKL)
    if not isinstance(coords_df, pd.DataFrame):
        raise TypeError(f"{COORDS_PKL} is not a pandas DataFrame")
    if not isinstance(geo_df, pd.DataFrame):
        raise TypeError(f"{GEO_PKL} is not a pandas DataFrame")
    _loaded["coords_df"] = coords_df
    _loaded["geo_df"] = geo_df
    _loaded["loaded_at"] = time.time()
    return coords_df, geo_df


def get_frames() -> Tuple[pd.DataFrame, pd.DataFrame]:
    if _loaded["coords_df"] is None or _loaded["geo_df"] is None:
        return load_data()
    return _loaded["coords_df"], _loaded["geo_df"]


REQUIRED_COORD_COLS = {"lat_min", "lat_max", "lon_min", "lon_max", "name"}
GEO_STATE_COL = "State"


def find_state(lat: float, lon: float, df: pd.DataFrame) -> Optional[str]:
    missing = REQUIRED_COORD_COLS - set(df.columns)
    if missing:
        raise KeyError(f"coordinates_data is missing required columns: {sorted(missing)}")
    for _, row in df.iterrows():
        if (row["lat_min"] <= lat <= row["lat_max"]) and (row["lon_min"] <= lon <= row["lon_max"]):
            return str(row["name"])
    return None


def get_geological_info(state: str, geo_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    if GEO_STATE_COL not in geo_df.columns:
        raise KeyError(f"geological_data missing '{GEO_STATE_COL}' column")
    data = geo_df[geo_df[GEO_STATE_COL] == state]
    if not data.empty:
        record = data.iloc[0].to_dict()
        return {str(k): (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v)) for k, v in record.items()}
    return None


def generate_synthetic_terrain(elevation: float, grid_size: int = 50, variation: float = 50.0):
    base = float(elevation)
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    xv, yv = np.meshgrid(x, y)
    noise = (np.random.rand(grid_size, grid_size) - 0.5) * variation
    z = base + noise
    return xv, yv, z


def render_terrain_png(x, y, z) -> bytes:
    try:
        import plotly.graph_objects as go
        fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
        fig.update_layout(
            title="Synthetic 3D Terrain",
            autosize=False,
            width=700, height=700,
            scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Elevation (m)"),
            margin=dict(l=0, r=0, t=30, b=0),
        )
        png_bytes = fig.to_image(format="png")
        return png_bytes
    except Exception:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(x, y, z, linewidth=0, antialiased=True)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Elevation (m)")
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf.read()



@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    lat = payload.get("latitude")
    lon = payload.get("longitude")

    if lat is None or lon is None:
        return jsonify({"error": "latitude and longitude are required"}), 400

    try:
        lat = float(lat)
        lon = float(lon)
    except Exception:
        return jsonify({"error": "latitude/longitude must be numeric"}), 400

    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return jsonify({"error": "latitude must be in [-90,90], longitude in [-180,180]"}), 400

    coords_df, geo_df = get_frames()
    state = find_state(lat, lon, coords_df)
    if not state:
        return jsonify({"state": None, "message": "No matching state for given coordinates"}), 404

    geo_info = get_geological_info(state, geo_df)

    include_img = bool(payload.get("include_terrain_png", False))
    grid_size = int(payload.get("grid_size", 50))
    variation = float(payload.get("variation", 50.0))
    resp = {
        "state": state,
        "geology": geo_info,
    }

    if include_img and geo_info and "Average_Elevation_m" in geo_info and geo_info["Average_Elevation_m"] is not None:
        x, y, z = generate_synthetic_terrain(geo_info["Average_Elevation_m"], grid_size=grid_size, variation=variation)
        png = render_terrain_png(x, y, z)
        resp["terrain_png_b64"] = base64.b64encode(png).decode("ascii")

    return jsonify(resp)


@app.get("/terrain")
def terrain_png():
    try:
        lat = float(request.args.get("lat"))
        lon = float(request.args.get("lon"))
    except Exception:
        abort(400, "lat and lon query params required and must be numeric")

    coords_df, geo_df = get_frames()
    state = find_state(lat, lon, coords_df)
    if not state:
        abort(404, "No matching state for given coordinates")

    geo_info = get_geological_info(state, geo_df)
    if not geo_info or "Average_Elevation_m" not in geo_info or geo_info["Average_Elevation_m"] is None:
        abort(404, "No geological elevation info to generate terrain")

    grid_size = int(request.args.get("grid_size", 50))
    variation = float(request.args.get("variation", 50.0))
    x, y, z = generate_synthetic_terrain(geo_info["Average_Elevation_m"], grid_size=grid_size, variation=variation)
    png = render_terrain_png(x, y, z)
    return send_file(io.BytesIO(png), mimetype="image/png", download_name=f"{state}_terrain.png")


if __name__ == "__main__":
    load_data()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)), debug=True)
