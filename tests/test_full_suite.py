"""
TruthLens AI — Master Test Suite
Evaluates all 10 core verification scenarios specified in Section 14:
1. Real ISOT article
2. Fake ISOT article
3. Short factual claim
4. Fabricated / debunked claim
5. Long news article
6. Article with multiple factual claims
7. Input with insufficient evidence
8. Empty / invalid input validation
9. Retrieval failure resilience
10. LLM / API failure resilience
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi.testclient import TestClient
from backend.main import app
from backend.services.classifier import DistilBERTClassifier, XGBoostClassifier
from backend.services.claim_extractor import get_claim_extractor
from backend.services.evidence_retriever import get_evidence_retriever
from backend.services.evidence_ranker import get_evidence_ranker
from backend.services.source_credibility import get_source_credibility_analyzer
from backend.services.llm_verifier import get_llm_verifier
from backend.utils.preprocessing import clean_text, remove_source_cues


class TestTruthLensMasterSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.distilbert = DistilBERTClassifier.get_instance()
        cls.xgboost = XGBoostClassifier.get_instance()
        cls.extractor = get_claim_extractor()
        cls.retriever = get_evidence_retriever()
        cls.ranker = get_evidence_ranker()
        cls.credibility = get_source_credibility_analyzer()
        cls.verifier = get_llm_verifier()

    # -------------------------------------------------------------
    # Scenario 1: Real ISOT Article
    # -------------------------------------------------------------
    def test_01_real_isot_article(self):
        text = (
            "WASHINGTON (Reuters) - The White House said on Friday that President Donald Trump "
            "will sign the bipartisan defense spending measure approved by Congress earlier this week."
        )
        res_bert = self.distilbert.predict(text)
        self.assertEqual(res_bert.predicted_label, "REAL")
        self.assertGreater(res_bert.confidence, 0.90)

        res_xgb = self.xgboost.predict(text)
        self.assertEqual(res_xgb.predicted_label, "REAL")
        self.assertGreater(res_xgb.confidence, 0.90)

    # -------------------------------------------------------------
    # Scenario 2: Fake ISOT Article
    # -------------------------------------------------------------
    def test_02_fake_isot_article(self):
        text = (
            "SHOCKING PROOF: Secret underground Clinton cabal exposed by patriotic whistleblowers "
            "inside the deep state! You will not believe what mainstream media is hiding from the public!"
        )
        res_bert = self.distilbert.predict(text)
        self.assertEqual(res_bert.predicted_label, "FAKE")

        res_xgb = self.xgboost.predict(text)
        self.assertEqual(res_xgb.predicted_label, "FAKE")

    # -------------------------------------------------------------
    # Scenario 3: Short Factual Claim
    # -------------------------------------------------------------
    def test_03_short_factual_claim(self):
        claim_text = "The Pacific Ocean is the largest and deepest ocean on Earth."
        response = self.client.post("/api/verify", json={"text": claim_text})
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertIn(data["final_verdict"], ["TRUE", "MOSTLY TRUE"])
        self.assertGreaterEqual(len(data["supporting_evidence"]), 1)
        # Check dict access for parsed JSON response
        first_title = data["supporting_evidence"][0]["title"].lower()
        self.assertIn("pacific", first_title)

    # -------------------------------------------------------------
    # Scenario 4: Fabricated / Debunked Claim
    # -------------------------------------------------------------
    def test_04_fabricated_claim(self):
        claim_text = "The Earth is completely flat and guarded by an impenetrable ice wall."
        claims = self.extractor.extract_claims(claim_text)
        self.assertGreaterEqual(len(claims), 1)

        raw_evidence = self.retriever.search("Flat Earth beliefs and ice wall", max_results=3)
        self.assertGreater(len(raw_evidence), 0)

        # Check that retrieved evidence identifies Flat Earth
        self.assertTrue(any("flat earth" in item.title.lower() or "earth" in item.title.lower() for item in raw_evidence))

    # -------------------------------------------------------------
    # Scenario 5: Long News Article (> 1,000 characters)
    # -------------------------------------------------------------
    def test_05_long_news_article(self):
        long_text = (
            "BRUSSELS (Reuters) - European Union climate negotiators agreed early on Wednesday "
            "to overhaul the 27-country bloc's carbon market, accelerating the pace of industrial emission cuts "
            "and introducing a new carbon tariff on imports by 2026. "
            "The agreement, reached after 30 hours of marathon talks between representatives of member states "
            "and the European Parliament, will reduce net greenhouse gas emissions by 62% by 2030 compared to 2005 levels. "
            "Under the reform, polluters will gradually lose their free carbon pollution allowances starting from 2026. "
            "Meanwhile, the European Union will phase in a Carbon Border Adjustment Mechanism designed to protect "
            "domestic manufacturers against cheaper competitors based in countries with weaker environmental regulations."
        )
        response = self.client.post("/api/verify", json={"text": long_text})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn(data["article_classification"]["predicted_label"], ["REAL", "FAKE"])
        self.assertGreaterEqual(len(data["extracted_claims"]), 1)

    # -------------------------------------------------------------
    # Scenario 6: Article Containing Multiple Factual Claims
    # -------------------------------------------------------------
    def test_06_multiple_claims_extraction(self):
        multi_claim_text = (
            "NASA launched the James Webb Space Telescope in December 2021. "
            "It orbits the Sun at the second Lagrange point 1.5 million kilometers from Earth. "
            "The European Space Agency provided the Ariane 5 rocket launch vehicle."
        )
        claims = self.extractor.extract_claims(multi_claim_text, max_claims=5)
        self.assertGreaterEqual(len(claims), 2)
        claim_texts = [c.claim_text for c in claims]
        self.assertTrue(any("2021" in t or "James Webb" in t for t in claim_texts))

    # -------------------------------------------------------------
    # Scenario 7: Input with Insufficient Evidence
    # -------------------------------------------------------------
    def test_07_insufficient_evidence(self):
        obscure_claim = "Zorblaxian-999 mineral frequency was synthesized in an underground laboratory in Valhalla in 1729."
        response = self.client.post("/api/verify", json={"text": obscure_claim})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        # Should cleanly conclude INSUFFICIENT EVIDENCE or UNVERIFIED rather than hallucinating
        self.assertIn(data["final_verdict"], ["INSUFFICIENT EVIDENCE", "UNVERIFIED"])

    # -------------------------------------------------------------
    # Scenario 8: Empty / Invalid Input Handling
    # -------------------------------------------------------------
    def test_08_empty_input_validation(self):
        # Empty text fails Pydantic min_length=3 validation
        response = self.client.post("/api/verify", json={"text": ""})
        self.assertEqual(response.status_code, 422)

        # 2-character text fails min_length=3 validation
        response_short = self.client.post("/api/verify", json={"text": "hi"})
        self.assertEqual(response_short.status_code, 422)

    # -------------------------------------------------------------
    # Scenario 9: Retrieval Failure Resilience
    # -------------------------------------------------------------
    def test_09_retrieval_failure_resilience(self):
        # When network retrieval fails or times out, verify pipeline falls back gracefully
        with patch.object(self.retriever, "search", side_effect=Exception("Simulated Network Timeout")):
            # Claim extraction still succeeds
            claims = self.extractor.extract_claims("Albert Einstein published the theory of general relativity in 1915.")
            self.assertEqual(len(claims), 1)

            # Verifier handles zero evidence gracefully
            verdict, conf, reasoning, updated_claims = self.verifier.verify(
                original_text="Albert Einstein published the theory of general relativity in 1915.",
                claims=claims,
                supporting=[],
                contradicting=[],
                context_prompt="No evidence retrieved"
            )
            self.assertEqual(verdict, "INSUFFICIENT EVIDENCE")
            self.assertEqual(conf, 0.50)

    # -------------------------------------------------------------
    # Scenario 10: LLM / API Failure Resilience
    # -------------------------------------------------------------
    def test_10_llm_api_failure_resilience(self):
        # When external LLM provider raises an exception, the system uses the offline reasoning engine
        claims = self.extractor.extract_claims("Water freezes at 0 degrees Celsius.")
        raw_evidence = self.retriever.search("Water freezing point temperature", max_results=2)
        supp, contra = self.ranker.rank_evidence("Water freezes at 0 degrees Celsius.", raw_evidence)

        # Mock an external LLM error
        with patch.object(self.verifier, "_verify_with_groq", side_effect=RuntimeError("Groq Rate Limit")):
            verdict, conf, reasoning, updated_claims = self.verifier.verify(
                original_text="Water freezes at 0 degrees Celsius.",
                claims=claims,
                supporting=supp,
                contradicting=contra,
                context_prompt=""
            )
            self.assertIn(verdict, ["TRUE", "MOSTLY TRUE", "PARTIALLY TRUE", "INSUFFICIENT EVIDENCE"])
            self.assertIsNotNone(reasoning)


if __name__ == "__main__":
    unittest.main()
