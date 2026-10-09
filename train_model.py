from pathlib import Path
import gzip
import json
import os
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
    ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "artifacts"
DATA_PATH = DATA_DIR / "creditcard.csv"
GZ_PATH = DATA_DIR / "creditcard.csv.gz"
RANDOM_STATE = 42
TARGET = "Class"

def load_dataset():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    if GZ_PATH.exists():
        with gzip.open(GZ_PATH, "rt") as f:
            return pd.read_csv(f)
    raise FileNotFoundError(
        "Dataset not found. Put creditcard.csv or creditcard.csv.gz in the data/ directory."
    )

def train_model():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading credit-card transactions dataset...")
    data = load_dataset()
    if TARGET not in data.columns:
        raise ValueError("Dataset must contain the target column 'Class'.")
    if data[TARGET].isna().any():
        raise ValueError("Target column contains missing values.")
    if set(data[TARGET].unique()) - {0, 1}:
        raise ValueError("Class must contain only 0 (legitimate) and 1 (fraud).")

    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    X = data.drop(columns=[TARGET])
    y = data[TARGET].astype(int)

    if y.nunique() != 2:
        raise ValueError("Both classes (0 and 1) must be present.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Records: {len(data)}")
    print(f"Features: {X.shape[1]}")
    print(f"Training records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")
    print(f"Fraud cases: {int(y.sum())}; legitimate cases: {int((y == 0).sum())}")

    # class_weight='balanced' compensates for the severe class imbalance.
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ))
    ])
    print("Training Logistic Regression...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    metrics = {
        "model": "LogisticRegression",
        "dataset": "creditcard.csv",
        "records": int(len(data)),
        "features": int(X.shape[1]),
        "training_records": int(len(X_train)),
        "testing_records": int(len(X_test)),
        "fraud_records": int(y.sum()),
        "legitimate_records": int((y == 0).sum()),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "pr_auc_average_precision": float(average_precision_score(y_test, probabilities)),
        "confusion_matrix_labels": ["legitimate (0)", "fraud (1)"],
        "confusion_matrix": cm.tolist(),
        "random_state": RANDOM_STATE,
        "test_size": 0.20
    }

    print("\nEvaluation metrics")
    for key in ["accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc_average_precision"]:
        print(f"{key}: {metrics[key]:.6f}")
    print("\nConfusion matrix (rows=actual, columns=predicted; labels [0, 1]):")
    print(cm)
    print("\nClassification report:")
    print(classification_report(y_test, predictions, labels=[0, 1],
                                target_names=["legitimate", "fraud"], zero_division=0))

    joblib.dump(model, OUT_DIR / "fraud_detection_model.pkl")
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (OUT_DIR / "classification_report.txt").write_text(
        classification_report(y_test, predictions, labels=[0, 1],
                              target_names=["legitimate", "fraud"], zero_division=0)
    )

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legitimate", "Fraud"])
    disp.plot(values_format="d")
    plt.title("Credit Card Fraud Detection - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "confusion_matrix.png", dpi=160)
    plt.close()

    RocCurveDisplay.from_predictions(y_test, probabilities)
    plt.title("Credit Card Fraud Detection - ROC Curve")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "roc_curve.png", dpi=160)
    plt.close()

    PrecisionRecallDisplay.from_predictions(y_test, probabilities)
    plt.title("Credit Card Fraud Detection - Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "precision_recall_curve.png", dpi=160)
    plt.close()

    print(f"\nSaved model and metrics in: {OUT_DIR}")
    return metrics

if __name__ == "__main__":
    train_model()
