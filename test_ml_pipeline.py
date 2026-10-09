import json
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts"

class TestFraudDetectionPipeline(unittest.TestCase):
    def test_model_artifact_exists(self):
        self.assertTrue((OUT / "fraud_detection_model.pkl").exists())

    def test_metrics_file_exists(self):
        self.assertTrue((OUT / "metrics.json").exists())

    def test_confusion_matrix_image_exists(self):
        self.assertTrue((OUT / "confusion_matrix.png").exists())

    def test_metrics_are_valid(self):
        metrics = json.loads((OUT / "metrics.json").read_text())
        for key in ["accuracy", "precision", "recall", "f1_score", "roc_auc",
                    "pr_auc_average_precision"]:
            self.assertGreaterEqual(metrics[key], 0.0)
            self.assertLessEqual(metrics[key], 1.0)
        self.assertEqual(metrics["training_records"] + metrics["testing_records"],
                         metrics["records"])

    def test_confusion_matrix_shape(self):
        metrics = json.loads((OUT / "metrics.json").read_text())
        matrix = np.array(metrics["confusion_matrix"])
        self.assertEqual(matrix.shape, (2, 2))
        self.assertEqual(int(matrix.sum()), metrics["testing_records"])

    def test_saved_model_predicts_valid_classes(self):
        model = joblib.load(OUT / "fraud_detection_model.pkl")
        sample = pd.DataFrame([{
            "Time": 0, "V1": -1.359807, "V2": -0.072781, "V3": 2.536347,
            "V4": 1.378155, "V5": -0.338321, "V6": 0.462388, "V7": 0.239599,
            "V8": 0.098698, "V9": 0.363787, "V10": 0.090794, "V11": -0.551600,
            "V12": -0.617801, "V13": -0.991390, "V14": -0.311169, "V15": 1.468177,
            "V16": -0.470401, "V17": 0.207971, "V18": 0.025791, "V19": 0.403993,
            "V20": 0.251412, "V21": -0.018307, "V22": 0.277838, "V23": -0.110474,
            "V24": 0.066928, "V25": 0.128539, "V26": -0.189115, "V27": 0.133558,
            "V28": -0.021053, "Amount": 149.62
        }])
        prediction = int(model.predict(sample)[0])
        self.assertIn(prediction, [0, 1])

if __name__ == "__main__":
    unittest.main(verbosity=2)
