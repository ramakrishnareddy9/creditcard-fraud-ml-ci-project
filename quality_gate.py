
import json
import sys
from pathlib import Path

MINIMUM_ACCURACY = 0.85
METRICS_FILE = Path("artifacts/metrics.json")


def main():
    print("Reading model evaluation metrics...")

    if not METRICS_FILE.exists():
        print("QUALITY GATE FAILED")
        print(f"Metrics file not found: {METRICS_FILE}")
        sys.exit(1)

    try:
        with METRICS_FILE.open("r", encoding="utf-8") as file:
            metrics = json.load(file)

        accuracy = float(metrics["accuracy"])

    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        print("QUALITY GATE FAILED")
        print(f"Unable to read a valid accuracy value: {error}")
        sys.exit(1)

    if not 0.0 <= accuracy <= 1.0:
        print("QUALITY GATE FAILED")
        print("Accuracy must be between 0 and 1.")
        sys.exit(1)

    print(f"Model Accuracy   : {accuracy:.4f}")
    print(f"Required Accuracy: {MINIMUM_ACCURACY:.2f}")

    if accuracy < MINIMUM_ACCURACY:
        print("QUALITY GATE FAILED")
        print("Model performance is below the required threshold.")
        sys.exit(1)

    print("QUALITY GATE PASSED")
    print("Model performance satisfies the required threshold.")
    sys.exit(0)


if __name__ == "__main__":
    main()
