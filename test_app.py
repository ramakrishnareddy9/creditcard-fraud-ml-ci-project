
import unittest

from app import app, load_model


class TestFraudPredictionAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        load_model()

    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["model_loaded"])

    def test_prediction_endpoint(self):
        features = {"Time": 0, "Amount": 100}
        features.update({f"V{i}": 0 for i in range(1, 29)})

        response = self.client.post(
            "/predict",
            json={"features": features}
        )

        self.assertEqual(response.status_code, 200)
        result = response.get_json()

        self.assertIn(result["prediction"], [0, 1])
        self.assertIn(result["label"], ["legitimate", "fraud"])
        self.assertGreaterEqual(result["fraud_probability"], 0)
        self.assertLessEqual(result["fraud_probability"], 1)

    def test_missing_features_are_rejected(self):
        response = self.client.post(
            "/predict",
            json={"features": {"Time": 0}}
        )
        self.assertEqual(response.status_code, 400)

    def test_non_numeric_features_are_rejected(self):
        features = {"Time": 0, "Amount": 100}
        features.update({f"V{i}": 0 for i in range(1, 29)})
        features["V1"] = "not-a-number"

        response = self.client.post(
            "/predict",
            json={"features": features}
        )
        self.assertEqual(response.status_code, 400)

    def test_non_json_request_is_rejected(self):
        response = self.client.post(
            "/predict",
            data="invalid",
            content_type="text/plain"
        )
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
