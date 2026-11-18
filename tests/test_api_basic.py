import unittest
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi.testclient import TestClient
from backend.main import app


class TestApiPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_root_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["project"], "TruthLens AI")
        self.assertEqual(data["status"], "online")

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["distilbert_loaded"])
        self.assertTrue(data["xgboost_available"])

    def test_classify_distilbert(self):
        payload = {
            "text": "WASHINGTON (Reuters) - The White House confirmed on Friday that international negotiations will proceed.",
            "model": "distilbert"
        }
        response = self.client.post("/api/classify", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["predicted_label"], "REAL")
        self.assertGreaterEqual(data["confidence"], 0.90)

    def test_classify_xgboost(self):
        payload = {
            "text": "WASHINGTON (Reuters) - The White House confirmed on Friday that international negotiations will proceed.",
            "model": "xgboost"
        }
        response = self.client.post("/api/classify", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["predicted_label"], "REAL")
        self.assertGreaterEqual(data["confidence"], 0.90)

    def test_extract_claims(self):
        payload = {
            "text": "NASA launched the James Webb Space Telescope in December 2021. It orbits the Sun at Lagrange point 2."
        }
        response = self.client.post("/api/extract-claims", json=payload)
        self.assertEqual(response.status_code, 200)
        claims = response.json()
        self.assertGreaterEqual(len(claims), 1)
        self.assertIn("claim_text", claims[0])

    def test_full_verify_pipeline(self):
        payload = {
            "text": "NASA launched the James Webb Space Telescope in December 2021.",
            "include_evidence": True,
            "llm_reasoning": True
        }
        response = self.client.post("/api/verify", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Check legacy compatibility fields
        self.assertIn("verdict", data)
        self.assertIn(data["verdict"], ["REAL", "FAKE"])
        self.assertIn("confidence", data)

        # Check full verification report fields
        self.assertIn("final_verdict", data)
        self.assertIn(data["final_verdict"], [
            "TRUE", "MOSTLY TRUE", "PARTIALLY TRUE", "MOSTLY FALSE", "FALSE", "UNVERIFIED", "INSUFFICIENT EVIDENCE"
        ])
        self.assertIn("article_classification", data)
        self.assertIn("extracted_claims", data)
        self.assertIn("source_credibility", data)
        self.assertIn("pipeline_steps", data)
        self.assertGreater(len(data["pipeline_steps"]), 0)


if __name__ == "__main__":
    unittest.main()
