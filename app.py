
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, request

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "fraud_detection_model.pkl"

FEATURE_NAMES = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount"]
)

model = None


def load_model():
    global model
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. Run train_model.py first."
        )
    model = joblib.load(MODEL_PATH)


@app.get("/health")
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    }), 200


@app.post("/predict")
def predict():
    if model is None:
        return jsonify({"error": "Model is not loaded"}), 503

    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        return jsonify({"error": "Send a JSON object"}), 400

    features = payload.get("features")

    if not isinstance(features, dict):
        return jsonify({
            "error": "The 'features' field must be a JSON object"
        }), 400

    missing = [name for name in FEATURE_NAMES if name not in features]
    extra = [name for name in features if name not in FEATURE_NAMES]

    if missing or extra:
        return jsonify({
            "error": "Incorrect feature names",
            "missing_features": missing,
            "unexpected_features": extra
        }), 400

    try:
        values = [float(features[name]) for name in FEATURE_NAMES]
    except (TypeError, ValueError):
        return jsonify({
            "error": "All feature values must be numeric"
        }), 400

    if not np.isfinite(values).all():
        return jsonify({
            "error": "Feature values must be finite numbers"
        }), 400

    prediction = int(model.predict([values])[0])
    probabilities = model.predict_proba([values])[0]

    return jsonify({
        "prediction": prediction,
        "label": "fraud" if prediction == 1 else "legitimate",
        "fraud_probability": float(probabilities[1])
    }), 200


if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=5000, debug=False)