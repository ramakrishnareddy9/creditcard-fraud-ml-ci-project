
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DRIFT_DIR = ROOT / "drift"
REFERENCE = DRIFT_DIR / "reference_data.csv"
CURRENT = DRIFT_DIR / "current_data.csv"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    choices=["stable", "drift"],
    default="stable",
)
parser.add_argument("--rows", type=int, default=500)
args = parser.parse_args()

if not REFERENCE.exists():
    raise FileNotFoundError(
        "Run prepare_reference.py before simulating current data."
    )
if args.rows < 2:
    raise ValueError("--rows must be at least 2")

reference = pd.read_csv(REFERENCE)
if reference.empty:
    raise ValueError("Reference dataset is empty.")

rng = np.random.default_rng(42)
n = min(args.rows, len(reference))

current = reference.sample(
    n=n,
    replace=len(reference) < n,
    random_state=42,
).reset_index(drop=True)

# Small noise for the classroom demonstration.
for feature in current.columns:
    values = current[feature].to_numpy(dtype=float)
    scale = float(np.nanstd(reference[feature]))
    noise_scale = max(scale * 0.01, 1e-9)
    current[feature] = values + rng.normal(
        0, noise_scale, size=len(current)
    )

if args.mode == "drift":
    # Synthetic shift applied only to selected input features.
    for feature in ["Amount", "V1", "V2", "V3"]:
        if feature in current.columns:
            scale = float(np.nanstd(reference[feature]))
            current[feature] += max(scale, 1.0) * 2.0

current.to_csv(CURRENT, index=False)

print("Mode:", args.mode)
print("Rows:", len(current))
print("File:", CURRENT)
