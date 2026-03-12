"""
FastAPI inference app — proxies predictions from the MLflow model server.

Browser:  http://localhost:5002
JSON API: POST http://localhost:5002/predict

Requires MODEL_SERVER_URL env var (defaults to http://model-server:5001/invocations).
"""

import os
import requests
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI(title="Wine Quality Predictor")

MODEL_SERVER_URL = os.environ.get("MODEL_SERVER_URL", "http://model-server:5001/invocations")

# ── Payloads ────────────────────────────────────────────────────

# High quality sample (white wine, high alcohol, balanced acidity)
high_quality = {
    "dataframe_split": {
        "columns": [
            "fixed_acidity", "volatile_acidity", "citric_acid",
            "residual_sugar", "chlorides", "free_sulfur_dioxide",
            "total_sulfur_dioxide", "density", "pH", "sulphates",
            "alcohol", "is_red",
        ],
        "data": [[7.0, 0.25, 0.36, 1.6, 0.034, 30.0, 110.0, 0.9906, 3.24, 0.50, 12.8, 0]],
    }
}

# Low quality sample (red wine, high volatile acidity, extreme values)
low_quality = {
    "dataframe_split": {
        "columns": [
            "fixed_acidity", "volatile_acidity", "citric_acid",
            "residual_sugar", "chlorides", "free_sulfur_dioxide",
            "total_sulfur_dioxide", "density", "pH", "sulphates",
            "alcohol", "is_red",
        ],
        "data": [[7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4, 1]],
    }
}

# 1300 samples
# batch_payload = {"dataframe_split": X_test.to_dict(orient="split")}


def _predict(payload: dict) -> dict:
    """Send payload to MLflow model server and return probability + label."""
    resp = requests.post(MODEL_SERVER_URL, json=payload, timeout=10)
    resp.raise_for_status()  # raises exception on 4xx/5xx so errors don't fail silently
    prob = resp.json()["predictions"][0]
    return {
        "probability": round(prob, 4),
        "label": "High Quality" if prob >= 0.5 else "Not High Quality",
    }


# ── JSON API ────────────────────────────────────────────────────

@app.post("/predict")
def predict():
    return _predict(high_quality)


# ── Browser UI ──────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def ui():
    hq = _predict(high_quality)
    lq = _predict(low_quality)
    return f"""
    <html>
    <body style="font-family: system-ui, sans-serif; max-width: 600px; margin: 40px auto;">
      <h2>Wine Quality Predictions</h2>
      <p><b>High quality sample:</b> {hq['label']} (probability: {hq['probability']})</p>
      <p><b>Low quality sample:</b> {lq['label']} (probability: {lq['probability']})</p>
    </body>
    </html>
    """
