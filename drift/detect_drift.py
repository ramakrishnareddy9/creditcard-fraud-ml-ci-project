
from pathlib import Path
import json
import pandas as pd
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parents[1]
DRIFT_DIR = ROOT / "drift"
REFERENCE = DRIFT_DIR / "reference_data.csv"
CURRENT = DRIFT_DIR / "current_data.csv"
REPORT = DRIFT_DIR / "drift_report.json"

P_VALUE_THRESHOLD = 0.05
MIN_DRIFTED_FEATURES = 2

for path in [REFERENCE, CURRENT]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required input file missing: {path}"
        )

reference = pd.read_csv(REFERENCE)
current = pd.read_csv(CURRENT)

if list(reference.columns) != list(current.columns):
    raise ValueError(
        "Reference and current feature columns do not match."
    )

results = {}
drifted_features = []

for feature in reference.columns:
    ref = reference[feature].dropna()
    cur = current[feature].dropna()

    if ref.empty or cur.empty:
        raise ValueError(f"No usable values for {feature}")

    statistic, p_value = ks_2samp(ref, cur)

    detected = bool(p_value < P_VALUE_THRESHOLD)
    if detected:
        drifted_features.append(feature)

    results[feature] = {
        "ks_statistic": round(float(statistic), 6),
        "p_value": round(float(p_value), 8),
        "drift_detected": detected,
        "reference_mean": round(float(ref.mean()), 6),
        "current_mean": round(float(cur.mean()), 6),
    }

overall_drift = len(drifted_features) >= MIN_DRIFTED_FEATURES

report = {
    "method": "two-sample Kolmogorov-Smirnov test",
    "p_value_threshold": P_VALUE_THRESHOLD,
    "min_drifted_features": MIN_DRIFTED_FEATURES,
    "drifted_feature_count": len(drifted_features),
    "drifted_features": drifted_features,
    "overall_drift": overall_drift,
    "reference_file": str(REFERENCE.name),
    "current_file": str(CURRENT.name),
    "features": results,
}

REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
print("\nDrift report:", REPORT)
