# TruthLens AI — AI-Powered Fake News Detection & Evidence-Based Fact Verification

> **"TruthLens AI doesn't just tell you whether news is fake — it investigates the claims, finds the evidence, and explains why."**

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/ML-PyTorch%20%7C%20Transformers-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-15B8EC.svg?style=flat)](https://xgboost.readthedocs.io/)
[![React](https://img.shields.io/badge/Frontend-React%2019%20%2B%20Vite-61DAFB.svg?style=flat&logo=react)](https://vitejs.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 1. Project Overview

Conventional fake news detectors operate as black-box binary text classifiers. While effective at recognizing journalistic stylistic markers on dataset-specific corpora (such as Reuters vs. hyperpartisan blogs in the ISOT dataset), they **cannot verify empirical real-world factual claims**. 

**TruthLens AI** establishes a **dual-tier research architecture** that cleanly separates:
1. **Article Stylistic Classification:** Deep transformer (`distilbert-base-uncased`) and classical ML baselines (Logistic Regression, Naive Bayes, and **XGBoost**) to assess stylistic manipulation and writing patterns.
2. **Evidence-Based Factual Verification:** A Retrieval-Augmented Generation (RAG) pipeline that extracts verifiable factual propositions, queries multi-source live knowledge bases (Wikipedia and Web Search), ranks passages by lexical/semantic relevance, conducts stance detection, evaluates source credibility, and synthesizes an explainable verification report.

---

## 2. Multi-Stage Pipeline Architecture

```
                       USER INPUT (Headline / Article / Claim)
                                        │
                                        ▼
                            INPUT VALIDATION & SANITIZATION
                                        │
                 ┌──────────────────────┴──────────────────────┐
                 ▼                                             ▼
     ARTICLE CLASSIFICATION (Style)                FACT VERIFICATION (Truth)
                 │                                             │
      ┌──────────┴──────────┐                                  ▼
      │                     │                          CLAIM EXTRACTION
  DistilBERT             XGBoost                 (Verifiable Proposition Filter)
 (Checkpoint-7821)    (Hist 300 Trees)                         │
  Acc: 99.87%          Acc: 99.65%                             ▼
                                                       EVIDENCE RETRIEVAL
                                                 (Wikipedia + DuckDuckGo APIs)
                                                               │
                                                               ▼
                                                       EVIDENCE RANKING
                                                  (Semantic + Stance Detection)
                                                               │
                                                               ▼
                                                         RAG PIPELINE
                                                 (Prompt Injection Sandboxing)
                                                               │
                                                               ▼
                                                      LLM REASONING ENGINE
                                                 (Groq / OpenAI / Offline Fallback)
                                                               │
                                                               ▼
                                                  SOURCE CREDIBILITY ANALYSIS
                                                 (Domain Standing & Corroboration)
                                                               │
                 ┌─────────────────────────────────────────────┘
                 ▼
     SYNTHESIZED VERIFICATION DASHBOARD
     - Article Classification (Style Pattern)
     - Factual Verdict (TRUE, MOSTLY TRUE, PARTIALLY TRUE, FALSE, UNVERIFIED)
     - Extracted Claims Breakdown
     - Supporting & Contradicting Evidence Cards (with clickable URLs)
     - Source Reliability Tiers & Corroboration Score
     - Evidence-Grounded Reasoning & Disclaimers
```

---

## 3. Empirical Model Performance & Benchmarks

All models were evaluated on the held-out test split of the deduplicated **ISOT Fake News Dataset** (39,103 articles; 21,196 Real, 17,907 Fake; 80/20 stratified train/test split):

| Model Architecture | Feature Representation / Tokens | Accuracy | Precision | Recall | F1 Score | Latency (ms/sample) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **DistilBERT** (Fine-Tuned Checkpoint-7821) | Transformer Wordpiece (Max 256 tokens) | **99.87%** | **99.91%** | **99.86%** | **99.88%** | ~12.50 ms |
| **TF-IDF + XGBoost** (300 Hist Trees) | 50,000 N-Grams (1,2), Subsample 0.8 | **99.65%** | **99.51%** | **99.86%** | **99.68%** | **0.066 ms** |
| **TF-IDF + Logistic Regression** | 50,000 N-Grams (Title + Text) | 98.61% | 98.11% | 99.34% | 98.72% | 0.082 ms |
| **TF-IDF + Naive Bayes** | 50,000 N-Grams (Title + Text) | 95.69% | 96.01% | 96.04% | 96.03% | 0.035 ms |
| **TF-IDF + Logistic Regression** | Title Only | 94.23% | 93.38% | 96.18% | 94.76% | 0.040 ms |
| **TF-IDF + Logistic Regression** | Source Cues Removed (No "Reuters") | 98.17% | 97.67% | 98.99% | 98.32% | 0.080 ms |

> **Evaluation Metric Artifacts:** Saved in [`evaluation/baseline_results.csv`](file:///e:/Truthlens-AI/evaluation/baseline_results.csv) and [`evaluation/xgboost_evaluation.json`](file:///e:/Truthlens-AI/evaluation/xgboost_evaluation.json).

---

## 4. Repository Structure

```
TruthLens-AI/
├── backend/
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py              # Modular API endpoints (/api/verify, /api/classify, etc.)
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── request.py             # Pydantic v2 validation schemas
│   │   └── response.py            # Typed response schemas
│   ├── services/
│   │   ├── __init__.py
│   │   ├── classifier.py          # DistilBERT and XGBoost inference loaders
│   │   ├── claim_extractor.py     # Verifiable claim segmentation & filtering
│   │   ├── evidence_retriever.py  # Multi-source retrieval (Wikipedia, DuckDuckGo)
│   │   ├── evidence_ranker.py     # Stance detection & lexical/semantic ranking
│   │   ├── rag_pipeline.py        # Context prompt assembly & injection defense
│   │   ├── llm_verifier.py        # Groq/OpenAI reasoning + offline deterministic engine
│   │   └── source_credibility.py  # Domain standing registry & corroboration analysis
│   ├── utils/
│   │   ├── __init__.py
│   │   └── preprocessing.py       # Cleaning, source cue stripping, text truncation
│   ├── config.py                  # Dynamic path resolution & environment settings
│   ├── main.py                    # FastAPI entrypoint with CORS, logging & lifespan
│   ├── model_loader.py            # Backwards-compatible legacy inference wrapper
│   └── requirements.txt
├── data/
│   ├── processed/
│   │   └── truthlens_dataset.csv  # 39,103 cleaned articles (ISOT)
│   └── raw/
│       ├── Fake.csv
│       └── True.csv
├── evaluation/
│   ├── baseline_results.csv       # Comparative benchmark table
│   └── xgboost_evaluation.json   # Full confusion matrix & classification report
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── App.jsx                # Upgraded dark-mode verification dashboard
│   │   ├── App.css                # Polished UI with responsive layout
│   │   ├── main.jsx
│   │   └── index.css
│   ├── package.json
│   └── vite.config.js
├── models/
│   ├── distilbert_final_256/
│   │   └── checkpoint-7821/       # Fine-tuned PyTorch safetensors & tokenizer
│   ├── tfidf_vectorizer.joblib    # 50,000 n-gram fitted vectorizer
│   └── xgboost_model.joblib       # 300-tree trained histogram XGBoost model
├── notebooks/
│   ├── 01_EDA.ipynb               # Exploratory data analysis & baseline benchmarks
│   └── 02_DistilBERT.ipynb        # Fine-tuning pipeline & checkpoint evaluation
├── src/
│   ├── evaluate.py                # Benchmark runner & display utility
│   ├── preprocessing.py           # Pipeline preprocessing entrypoint
│   ├── train.py                   # Model training launcher
│   └── train_xgboost.py           # XGBoost training & evaluation script
├── tests/
│   ├── test_api_basic.py          # API route tests
│   └── test_full_suite.py         # 10 core verification scenario tests
├── .env.example                   # Environment configuration template
├── README.md                      # Comprehensive documentation
└── requirements.txt
```

---

## 5. Getting Started (Windows / PowerShell Guide)

### Prerequisites
- Python 3.11+
- Node.js 18+ & npm

### Step 1: Clone and Configure Environment
Open PowerShell in the project directory:
```powershell
cd E:\Truthlens-AI

# Create .env from template
Copy-Item .env.example .env
```

To configure optional external LLM reasoning (Groq or OpenAI), edit `.env`:
```ini
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```
*(Note: If no API keys are provided, TruthLens automatically runs its robust offline deterministic evidence reasoning engine).*

### Step 2: Install Python Dependencies
```powershell
py -3.11 -m pip install -r requirements.txt
```

### Step 3: Start the FastAPI Backend
```powershell
py -3.11 -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
The API is available at:
- **API Root:** `http://127.0.0.1:8000`
- **Swagger Interactive Docs:** `http://127.0.0.1:8000/docs`
- **System Health Endpoint:** `http://127.0.0.1:8000/health`

### Step 4: Start the React Frontend
In a separate terminal window:
```powershell
cd E:\Truthlens-AI\frontend
npm install
npm run dev
```
Open your browser at `http://localhost:5173`.

---

## 6. API Reference

### Health Check
`GET /health`
```json
{
  "status": "healthy",
  "project": "TruthLens AI",
  "version": "2.0.0",
  "distilbert_loaded": true,
  "xgboost_available": true,
  "llm_configured": false,
  "llm_provider": "offline",
  "retrieval_provider": "duckduckgo"
}
```

### Article Classification (Style Only)
`POST /api/classify`
```json
{
  "text": "WASHINGTON (Reuters) - The White House confirmed international talks.",
  "model": "distilbert" // or "xgboost"
}
```
**Response:**
```json
{
  "model_name": "DistilBERT (Fine-Tuned ISOT, max_len=256)",
  "predicted_label": "REAL",
  "confidence": 0.99999,
  "probabilities": {
    "FAKE": 0.00001,
    "REAL": 0.99999
  },
  "explanation": "DistilBERT transformer classification based on ISOT dataset style patterns."
}
```

### Full Verification Pipeline
`POST /api/verify` or `POST /verify`
```json
{
  "text": "NASA's James Webb Space Telescope detected atmospheric water vapor on exoplanet WASP-96b.",
  "classifier_model": "distilbert",
  "include_evidence": true,
  "llm_reasoning": true
}
```
**Response:**
```json
{
  "verdict": "REAL",
  "confidence": 0.999,
  "query": "NASA's James Webb Space Telescope detected atmospheric water vapor on exoplanet WASP-96b.",
  "final_verdict": "TRUE",
  "confidence_score": 0.95,
  "summary_reasoning": "Multiple authoritative references corroborate the assertion. For example, Wikipedia: James Webb Space Telescope confirms detection of exoplanetary atmospheric composition.",
  "article_classification": { ... },
  "extracted_claims": [
    {
      "id": 1,
      "claim_text": "NASA's James Webb Space Telescope detected atmospheric water vapor on exoplanet WASP-96b.",
      "is_verifiable": true,
      "verdict": "TRUE",
      "confidence": 0.95
    }
  ],
  "supporting_evidence": [
    {
      "id": 1,
      "title": "Wikipedia: James Webb Space Telescope",
      "snippet": "In July 2022, NASA revealed transit transmission spectra from WASP-96b showing clear signatures of water vapor...",
      "source_url": "https://en.wikipedia.org/wiki/James_Webb_Space_Telescope",
      "domain": "wikipedia.org",
      "relevance_score": 0.88
    }
  ],
  "contradicting_evidence": [],
  "source_credibility": {
    "overall_rating": "High Reliability",
    "corroboration_score": 0.90,
    "evaluated_sources": [
      {
        "domain": "wikipedia.org",
        "source_type": "Primary Wire Service / Established Fact-Checker",
        "credibility_tier": "High Reliability",
        "is_official_or_academic": false
      }
    ]
  },
  "pipeline_steps": [
    "Article Classification (DistilBERT)",
    "Claim Extraction (1 factual claim(s) identified)",
    "Evidence Retrieval (2 supporting, 0 contradicting)",
    "Evidence-Grounded Factual Reasoning",
    "Source Credibility Assessment"
  ],
  "execution_time_seconds": 1.28
}
```

---

## 7. Running the Automated Test Suite

TruthLens AI includes a master test suite validating all 10 core verification scenarios:
```powershell
py -3.11 tests/test_full_suite.py
```
**Scenarios Covered:**
1. Real ISOT article classification (Reuters style)
2. Fake ISOT article classification (Hyperpartisan clickbait)
3. Short factual empirical claim
4. Fabricated / debunked conspiracy claim
5. Long multi-paragraph news article (>1,000 characters)
6. Multi-claim extraction and segmentation
7. Synthetic claim with insufficient evidence (non-hallucination verification)
8. Empty and short input validation (Pydantic 422 handling)
9. Network / retrieval timeout failure resilience
10. External LLM API rate limit / outage fallback resilience

---

## 8. Known Limitations & Research Roadmap

- **Corpus Domain Shift:** The ISOT dataset consists primarily of 2016–2017 political news. The classical ML and DistilBERT models reflect this vocabulary. TruthLens mitigates this by emphasizing the RAG verification pipeline over raw classifier logits.
- **Search Latency:** Live search against public endpoints (Wikipedia and DuckDuckGo) introduces network round-trip overhead (typically 1–2 seconds). For production deployments, local vector store indexing (e.g. FAISS or ChromaDB on a verified news snapshot) will eliminate network variance.
- **Multi-Lingual Verification:** Current fine-tuned checkpoints and extraction regex patterns target English news articles. Extending claim extraction and embeddings to multilingual models is planned for TruthLens v3.

---

## 9. License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
#   T r u t h L e n s - A I  
 