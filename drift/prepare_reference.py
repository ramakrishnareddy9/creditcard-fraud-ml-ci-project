
from pathlib import Path
import gzip
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_GZ = ROOT / "data" / "creditcard.csv.gz"
DATA_CSV = ROOT / "data" / "creditcard.csv"
DRIFT_DIR = ROOT / "drift"
REFERENCE = DRIFT_DIR / "reference_data.csv"

DRIFT_DIR.mkdir(exist_ok=True)

if DATA_CSV.exists():
    df = pd.read_csv(DATA_CSV)
elif DATA_GZ.exists():
    with gzip.open(DATA_GZ, "rt") as f:
        df = pd.read_csv(f)
else:
    raise FileNotFoundError("Credit-card dataset not found.")

if "Class" not in df.columns:
    raise ValueError("Expected target column 'Class'.")

features = df.drop(columns=["Class"])
features = features.replace([float("inf"), float("-inf")], float("nan"))
features = features.dropna()

features.to_csv(REFERENCE, index=False)

print("Reference dataset created.")
print("Rows:", len(features))
print("Input features:", len(features.columns))
print("File:", REFERENCE)
