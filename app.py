
import logging
import os
import re
import sys
import time
import uuid
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, g, jsonify, request

# Configure logs for the terminal, Docker, and Render.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)

app = Flask(__name__)
app.logger.setLevel(logging.INFO)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "fraud_detection_model.pkl"

FEATURE_NAMES = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount"]
)

model = None


def load_model():
    """Load the trained model and log the outcome."""
    global model

    try:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"Model file not found: {MODEL_PATH}"
            )

        model = joblib.load(MODEL_PATH)
        app.logger.info(
            "event=model_loaded model_file=%s",
            MODEL_PATH.name,
        )
    except Exception:
        app.logger.exception(
            "event=model_load_failed model_file=%s",
            MODEL_PATH.name,
        )
        raise


@app.before_request
def start_request_logging():
    """Assign a request ID and record the start time."""
    supplied_id = request.headers.get("X-Request-ID", "")

    # Accept only short, safe request IDs; otherwise generate one.
    if supplied_id and re.fullmatch(r"[A-Za-z0-9_-]{1,64}", supplied_id):
        g.request_id = supplied_id
    else:
        g.request_id = uuid.uuid4().hex

    g.request_start_time = time.perf_counter()


@app.after_request
def finish_request_logging(response):
    """Log HTTP status and duration for every completed request."""
    request_id = getattr(g, "request_id", "unknown")
    start_time = getattr(g, "request_start_time", time.perf_counter())
    duration_ms = (time.perf_counter() - start_time) * 1000

    response.headers["X-Request-ID"] = request_id

    log_method = (
        app.logger.error if response.status_code >= 500
        else app.logger.warning if response.status_code >= 400
        else app.logger.info
    )

    log_method(
        "event=http_request request_id=%s method=%s path=%s "
        "status=%s duration_ms=%.2f",
        request_id,
        request.method,
        request.path,
        response.status_code,
        duration_ms,
    )

    return response


@app.get("/health")
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
    }), 200


@app.post("/predict")
def predict():
    request_id = getattr(g, "request_id", "unknown")

    if model is None:
        app.logger.error(
            "event=prediction_rejected reason=model_not_loaded request_id=%s",
            request_id,
        )
        return jsonify({"error": "Model is not loaded"}), 503

    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        app.logger.warning(
            "event=validation_error reason=invalid_json_object request_id=%s",
            request_id,
        )
        return jsonify({"error": "Send a JSON object"}), 400

    features = payload.get("features")

    if not isinstance(features, dict):
        app.logger.warning(
            "event=validation_error reason=invalid_features_object request_id=%s",
            request_id,
        )
        return jsonify({
            "error": "The 'features' field must be a JSON object"
        }), 400

    missing = [name for name in FEATURE_NAMES if name not in features]
    extra = [name for name in features if name not in FEATURE_NAMES]

    if missing or extra:
        app.logger.warning(
            "event=validation_error reason=incorrect_feature_names "
            "missing_count=%s unexpected_count=%s request_id=%s",
            len(missing),
            len(extra),
            request_id,
        )
        return jsonify({
            "error": "Incorrect feature names",
            "missing_features": missing,
            "unexpected_features": extra,
        }), 400

    try:
        values = [float(features[name]) for name in FEATURE_NAMES]
    except (TypeError, ValueError):
        app.logger.warning(
            "event=validation_error reason=non_numeric_features request_id=%s",
            request_id,
        )
        return jsonify({
            "error": "All feature values must be numeric"
        }), 400

    if not np.isfinite(values).all():
        app.logger.warning(
            "event=validation_error reason=non_finite_features request_id=%s",
            request_id,
        )
        return jsonify({
            "error": "Feature values must be finite numbers"
        }), 400

    # Keep the existing model inference and response schema.
    try:
        prediction = int(model.predict([values])[0])
        probabilities = model.predict_proba([values])[0]
        fraud_probability = float(probabilities[1])

        app.logger.info(
            "event=prediction_success request_id=%s prediction=%s",
            request_id,
            prediction,
        )

        return jsonify({
            "prediction": prediction,
            "label": "fraud" if prediction == 1 else "legitimate",
            "fraud_probability": fraud_probability,
        }), 200

    except Exception:
        app.logger.exception(
            "event=prediction_failed request_id=%s",
            request_id,
        )
        return jsonify({"error": "Prediction failed"}), 500


if __name__ == "__main__":
    load_model()
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
