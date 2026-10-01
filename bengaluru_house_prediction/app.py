"""
app.py
------
Flask REST API that serves the trained house-price prediction model.

Endpoints
  GET  /api/locations          – returns the list of known locations
  GET  /api/health             – health check
  POST /api/predict            – predict house price

POST /api/predict  expects JSON:
  {
    "location":   "Whitefield",
    "total_sqft": 1200,
    "bhk":        3,
    "bath":       2
  }

Returns JSON:
  {
    "predicted_price_lakhs": 85.43,
    "message": "Estimated price: ₹85.43 Lakhs"
  }
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")

# ─── Load artifacts at startup ───────────────────────────────────────────────
def load_artifacts():
    model_path   = os.path.join(ARTIFACTS_DIR, "model.pkl")
    columns_path = os.path.join(ARTIFACTS_DIR, "columns.json")
    locs_path    = os.path.join(ARTIFACTS_DIR, "locations.json")

    for p in (model_path, columns_path, locs_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Artifact not found: {p}\n"
                "Run `python train_model.py` first to generate model artifacts."
            )

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(columns_path) as f:
        data_columns = json.load(f)["data_columns"]

    with open(locs_path) as f:
        locations = json.load(f)["locations"]

    return model, data_columns, locations


MODEL, DATA_COLUMNS, LOCATIONS = load_artifacts()

# ─── Prediction helper ────────────────────────────────────────────────────────
def predict_price(location: str, total_sqft: float, bhk: int, bath: int) -> float:
    """Return predicted price in Lakhs."""
    x = pd.DataFrame(columns=DATA_COLUMNS, data=np.zeros((1, len(DATA_COLUMNS))))
    x["total_sqft"] = total_sqft
    x["bath"]       = bath
    x["bhk"]        = bhk

    loc_lower = location.strip().lower()
    for col in DATA_COLUMNS:
        if col.strip().lower() == loc_lower:
            x[col] = 1
            break

    return round(float(MODEL.predict(x)[0]), 2)


# ─── Routes ──────────────────────────────────────────────────────────────────
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_columns": len(DATA_COLUMNS)})


@app.route("/api/locations", methods=["GET"])
def get_locations():
    return jsonify({"locations": LOCATIONS})


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        location   = data.get("location", "")
        total_sqft = float(data.get("total_sqft", 0))
        bhk        = int(data.get("bhk", 0))
        bath       = int(data.get("bath", 0))

        # Basic validation
        errors = []
        if not location:
            errors.append("'location' is required.")
        if total_sqft <= 0:
            errors.append("'total_sqft' must be a positive number.")
        if bhk <= 0:
            errors.append("'bhk' must be a positive integer.")
        if bath <= 0:
            errors.append("'bath' must be a positive integer.")
        if errors:
            return jsonify({"error": " ".join(errors)}), 400

        price = predict_price(location, total_sqft, bhk, bath)

        if price < 0:
            return jsonify({"error": "Model returned an invalid prediction. Check your inputs."}), 422

        return jsonify({
            "predicted_price_lakhs": price,
            "message": f"Estimated price: ₹{price:.2f} Lakhs",
            "inputs": {
                "location":   location,
                "total_sqft": total_sqft,
                "bhk":        bhk,
                "bath":       bath,
            },
        })

    except (ValueError, TypeError) as exc:
        return jsonify({"error": f"Invalid input: {exc}"}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ─── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Starting Flask server on http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
