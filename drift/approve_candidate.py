
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
DRIFT_DIR = ROOT / "drift"
REPORT = DRIFT_DIR / "drift_report.json"
CURRENT = ROOT / "artifacts" / "metrics.json"
CANDIDATE = ROOT / "artifacts" / "candidate_metrics.json"

def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))

def main():
    drift = load_json(REPORT)
    current = load_json(CURRENT)
    candidate = load_json(CANDIDATE)

    if not drift.get("overall_drift", False):
        print("NOT APPROVED: No overall drift detected.")
        return 1

    if candidate.get("not_production_retraining", False):
        print("NOT APPROVED: Candidate is a demonstration model.")
        return 1

    required = ["precision", "recall", "f1_score", "pr_auc_average_precision"]
    if not all(key in candidate for key in required):
        print("NOT APPROVED: Candidate evaluation metrics are incomplete.")
        return 1

    if candidate.get("training_source") != "new_labeled_data":
        print("NOT APPROVED: Candidate is not trained on newly labeled data.")
        return 1

    if candidate.get("evaluation_source") != "independent_labeled_holdout":
        print("NOT APPROVED: Independent labeled holdout evaluation is required.")
        return 1

    # Avoid promoting a candidate that regresses on key fraud metrics.
    for metric in required:
        old_value = float(current[metric])
        new_value = float(candidate[metric])
        if new_value < old_value:
            print(
                f"NOT APPROVED: {metric} regressed "
                f"from {old_value:.6f} to {new_value:.6f}."
            )
            return 1

    print("APPROVED: Candidate passed the configured non-regression checks.")
    print("This approval does not deploy the model automatically.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
