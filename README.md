# Credit Card Fraud Detection — ML CI with GitHub Actions

This adapts Practical-2 (Student Result Prediction) to the supplied credit-card transaction dataset.

## Dataset
Expected columns: `Time`, `V1`–`V28`, `Amount`, and target `Class`.
- `Class = 0`: legitimate transaction
- `Class = 1`: fraudulent transaction

The supplied dataset contains 284,807 rows, 30 input features, and 492 fraud examples. Because it is highly imbalanced, the project reports precision, recall, F1, ROC-AUC, and average precision in addition to accuracy.

## Project structure
```text
creditcard-fraud-ml-ci/
├── .github/workflows/ml-ci.yml
├── data/creditcard.csv.gz
├── artifacts/                  # generated during training
├── train_model.py
├── test_ml_pipeline.py
├── requirements.txt
└── README.md
```

## Run locally
Python 3.11 is recommended.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
python train_model.py
python -m unittest discover -v
```

Outputs are created in `artifacts/`: trained model (`fraud_detection_model.pkl`), `metrics.json`, classification report, confusion matrix, ROC curve, and precision-recall curve.

## GitHub setup
1. Create a public repository named `creditcard-fraud-ml-ci`.
2. Upload all project files, including `data/creditcard.csv.gz`.
3. Commit to the `main` branch.
4. Open **Actions** and select **Credit Card Fraud Detection ML CI**.
5. Confirm the train/evaluate, tests, and artifact-upload steps complete successfully.
6. Download `fraud-detection-results` from the completed workflow run to inspect model and plots.

The provided compressed CSV is included to keep the dataset file below GitHub's usual 100 MB single-file limit. Do not commit both the uncompressed CSV and compressed copy.

## Model design
A stratified 80/20 train/test split is used with a fixed random seed. The pipeline standardizes features and trains Logistic Regression with `class_weight="balanced"` to account for the rare fraud class. The test suite checks output files, metric ranges, confusion-matrix consistency, and that the saved model can make a binary prediction.

## Important interpretation
A successful CI run only confirms that the pipeline and tests execute. It does not prove the model is production-ready. Fraud detection should prioritize fraud-class recall and precision-recall performance, and model thresholds should be selected based on business costs and validation data.


# Credit Card Fraud Detection — Release Package

## Contents

- `app.py`: Flask prediction API
- `fraud_detection_model.pkl`: Trained model
- `requirements.txt`: Python dependencies

## Run locally

Install Python 3.11 or a compatible Python version.

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Ensure the trained model is available at:

`artifacts/fraud_detection_model.pkl`

Start the API:

```bash
python app.py
```

Health check:

`GET http://127.0.0.1:5000/health`

Prediction endpoint:

`POST http://127.0.0.1:5000/predict`

Send a JSON object containing all 30 features under
the `features` key. See the project README for an example.

This package is intended for practical demonstration
and local testing, not production financial decisions.
