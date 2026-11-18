# TruthLens AI — AI-Powered Fake News Detection & Evidence-Based Fact Verification

> **"TruthLens AI doesn't just predict whether news looks fake — it extracts the claims, searches real-time live evidence, and explains why."**

[![Live Demo](https://img.shields.io/badge/Frontend-Vercel-black?style=for-the-badge&logo=vercel)](https://vercel.com)
[![Backend API](https://img.shields.io/badge/Backend-Render-46E3B7?style=for-the-badge&logo=render)](https://truthlens-ai-wag8.onrender.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![React 19](https://img.shields.io/badge/React_19-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

---

## Live Deployments & Endpoints

| Service | Hosting | Status | URL |
| :--- | :--- | :---: | :--- |
| **Web Dashboard** | **Vercel** | 🟢 Live | `https://<your-app>.vercel.app` |
| **FastAPI Backend** | **Render** | 🟢 Live | [`https://truthlens-ai-wag8.onrender.com`](https://truthlens-ai-wag8.onrender.com) |
| **Interactive Docs** | **Swagger UI** | 🟢 Live | [`https://truthlens-ai-wag8.onrender.com/docs`](https://truthlens-ai-wag8.onrender.com/docs) |
| **System Health** | **JSON Endpoint** | 🟢 Live | [`https://truthlens-ai-wag8.onrender.com/health`](https://truthlens-ai-wag8.onrender.com/health) |

---

## 1. Project Overview & Research Contribution

Conventional fake news detectors operate as black-box binary classifiers. While effective at recognizing dataset-specific journalistic markers (such as Reuters styling vs. hyperpartisan clickbait in benchmark datasets), they **cannot verify empirical real-world factual claims**. A completely fabricated lie written in formal journalistic tone will fool standard classifiers, while genuine breaking news written casually may be misclassified as fake.

**TruthLens AI** introduces a **dual-tier research architecture** that cleanly decouples:

1. **Article Stylistic Classification (The "Form"):** Deep transformer (`distilbert-base-uncased`) and classical machine learning baselines (**XGBoost**, Logistic Regression, Naive Bayes) evaluate writing syntax, manipulative phrasing, and ISOT dataset stylistic n-grams.
2. **Evidence-Based Fact Verification (The "Truth"):** A real-time Retrieval-Augmented Generation (RAG) pipeline that segments text into verifiable factual propositions, queries multi-source live knowledge bases (**Google News RSS**, **Wikipedia API**, and **Web Search**), computes semantic passage relevance, conducts stance detection, checks source credibility, and synthesizes an evidence-grounded verdict.

---

## 2. Multi-Stage Pipeline Architecture

```
                       USER INPUT (Headline / Article / Claim)
                                        │
                                        ▼
                            INPUT VALIDATION & SANITIZATION
                       (Strip Web Boilerplate / Normalization)
                                        │
                  ┌─────────────────────┴─────────────────────┐
                  ▼                                           ▼
      TIER 1: ARTICLE STYLE CLASSIFICATION        TIER 2: EVIDENCE-BASED FACT VERIFICATION
                  │                                           │
       ┌──────────┴──────────┐                                ▼
       │                     │                        CLAIM EXTRACTION
   DistilBERT             XGBoost               (Verifiable Proposition Filter)
  (Checkpoint-7821)    (Hist 300 Trees)                       │
   Acc: 99.87%          Acc: 99.65%                           ▼
                                                      EVIDENCE RETRIEVAL
                                             ┌────────────────┼────────────────┐
                                             ▼                ▼                ▼
                                        Google News       Wikipedia        DuckDuckGo
                                        (Live News)     (Encyclopedia)    (Web Search)
                                             └────────────────┬────────────────┘
                                                              │
                                                              ▼
                                                      EVIDENCE RANKING
                                               (Lexical + Semantic Overlap)
                                                              │
                                                              ▼
                                                      STANCE DETECTION
                                            (Supporting vs. Contradicting Cues)
                                                              │
                                                              ▼
                                                 SOURCE CREDIBILITY ANALYSIS
                                              (Domain Standing & Corroboration)
                                                              │
                                                              ▼
                                                     RAG REASONING ENGINE
                                            (Groq Llama-3.3 / OpenAI / Offline)
                                                              │
                  ┌───────────────────────────────────────────┘
                  ▼
      SYNTHESIZED VERIFICATION DASHBOARD
      ├── 1. Stylistic Prediction & Probabilities (DistilBERT / XGBoost)
      ├── 2. Factual Verdict (TRUE, MOSTLY TRUE, PARTIALLY TRUE, FALSE, INSUFFICIENT EVIDENCE)
      ├── 3. Extracted Propositions & Individual Claim Verdicts
      ├── 4. Supporting & Contradicting Evidence Cards (Clickable Direct Links)
      ├── 5. Evaluated Domains & Corroboration Score
      └── 6. Evidence-Grounded Reasoning & Explanation
```

---

## 3. Empirical Model Benchmarks

All models were evaluated on the held-out test split of the deduplicated **ISOT Fake News Dataset** (39,103 articles: 21,196 Real, 17,907 Fake; 80/20 stratified train/test split):

| Model Architecture | Feature Representation / Tokens | Accuracy | Precision | Recall | F1 Score | Latency (ms/sample) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **DistilBERT** (Fine-Tuned Checkpoint-7821) | Transformer WordPiece (Max 256 tokens) | **99.87%** | **99.91%** | **99.86%** | **99.88%** | ~12.50 ms |
| **TF-IDF + XGBoost** (300 Hist Trees) | 50,000 N-Grams (1,2), Subsample 0.8 | **99.65%** | **99.51%** | **99.86%** | **99.68%** | **0.066 ms** |
| **TF-IDF + Logistic Regression** | 50,000 N-Grams (Title + Text) | 98.61% | 98.11% | 99.34% | 98.72% | 0.082 ms |
| **TF-IDF + Naive Bayes** | 50,000 N-Grams (Title + Text) | 95.69% | 96.01% | 96.04% | 96.03% | 0.035 ms |
| **TF-IDF + Logistic Regression** | Title Only | 94.23% | 93.38% | 96.18% | 94.76% | 0.040 ms |
| **TF-IDF + Logistic Regression** | Source Cues Removed (No "Reuters" leaks) | 98.17% | 97.67% | 98.99% | 98.32% | 0.080 ms |

> **Evaluation Artifacts:** Stored in [`evaluation/baseline_results.csv`](evaluation/baseline_results.csv) and [`evaluation/xgboost_evaluation.json`](evaluation/xgboost_evaluation.json).

---

## 4. Key Capabilities & Features

* **Real-Time Live News Retrieval**: Integrates Google News RSS feeds to fetch breaking and regional news within minutes of publication across thousands of outlets (Reuters, AP, BBC, Indian Express, The Hindu, NDTV, CNN).
* **Deep Anti-Hallucination Guardrails**: Adheres to strict evidence-grounded rules. If reliable primary passages cannot be verified, the system outputs `INSUFFICIENT EVIDENCE` (50% neutral confidence) rather than fabricating sources.
* **Dual-Model Inference Switch**: Seamlessly toggle between fine-tuned **DistilBERT** (deep transformer representation) and **XGBoost** (ultra-low latency sub-millisecond classifier) directly from the UI.
* **Entity-Aware Multi-Hop Search**: Intelligent query cleaner that strips copy-paste web noise (`Advertisement`, `Breaking:`, `Sponsored`) and extracts capitalized named entities to ensure maximum search hit rates.
* **Automated Stance Detection**: Identifies contradiction tokens (*hoax*, *debunked*, *refuted*, *denied*, *fabricated*) versus corroboration tokens (*confirmed*, *official*, *recorded*).
* **Independent Source Credibility Engine**: Classifies retrieved sources into credibility tiers (*Official/Institutional*, *Reputable News*, *General Web*, *Low Reliability*) and computes cross-source corroboration percentages.
* **Modern Responsive UI**: Built with React 19, Vite, and Lucide icons, featuring dark mode, glassmorphism cards, interactive sample prompts, expandable claim breakdowns, and clickable reference citations.

---

## 5. Repository Structure

```
TruthLens-AI/
├── backend/
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py              # Modular FastAPI endpoints (/api/verify, /api/classify, etc.)
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── request.py             # Pydantic v2 input validation models
│   │   └── response.py            # Typed response models with Swagger documentation
│   ├── services/
│   │   ├── __init__.py
│   │   ├── classifier.py          # DistilBERT and XGBoost singleton model loaders
│   │   ├── claim_extractor.py     # Proposition segmentation & factual filter
│   │   ├── evidence_retriever.py  # Google News RSS, Wikipedia API & DuckDuckGo retriever
│   │   ├── evidence_ranker.py     # Lexical/semantic relevance & stance detection
│   │   ├── llm_verifier.py        # Groq (Llama-3.3), OpenAI, & Offline deterministic engines
│   │   ├── rag_pipeline.py        # Sandboxed context prompt assembly
│   │   └── source_credibility.py  # Domain reliability registry & corroboration analyzer
│   ├── utils/
│   │   ├── __init__.py
│   │   └── preprocessing.py       # Text cleaning, boilerplate stripping & normalization
│   ├── config.py                  # Dynamic path resolution & environment configuration
│   └── main.py                    # FastAPI entrypoint with CORS, logging & lifespan handler
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── App.jsx                # Full-featured interactive verification dashboard
│   │   ├── App.css                # Polished responsive dark-mode styling
│   │   ├── main.jsx
│   │   └── index.css
│   ├── .env.example
│   ├── package.json
│   ├── vercel.json                # Single-page application route rewrite configuration
│   └── vite.config.js
├── models/
│   ├── distilbert_final_256/
│   │   └── checkpoint-7821/       # Fine-tuned PyTorch safetensors, tokenizer & configs
│   ├── tfidf_vectorizer.joblib    # 50,000 n-gram fitted TF-IDF vectorizer
│   └── xgboost_model.joblib       # 300-tree trained histogram XGBoost model
├── evaluation/
│   ├── baseline_results.csv       # Comparative baseline metrics
│   └── xgboost_evaluation.json   # Full confusion matrix & classification report
├── notebooks/
│   ├── 01_EDA.ipynb               # Exploratory data analysis & baseline benchmarks
│   └── 02_DistilBERT.ipynb        # Fine-tuning pipeline & checkpoint evaluation
├── src/
│   ├── evaluate.py                # Benchmark runner & display utility
│   ├── preprocessing.py           # Dataset preprocessing entrypoint
│   ├── train.py                   # Model training launcher
│   └── train_xgboost.py           # XGBoost training & evaluation script
├── tests/
│   ├── test_api_basic.py          # API route health & classification tests
│   └── test_full_suite.py         # 10 comprehensive verification scenario tests
├── .dockerignore
├── .gitignore
├── Dockerfile                     # Production CPU-optimized Docker container definition
├── requirements.txt               # Backend Python dependencies
└── README.md
```

---

## 6. Quickstart & Local Setup

### Prerequisites
* **Python 3.11+**
* **Node.js 18+ & npm**
* **Git & Git LFS** (for models > 100 MB)

### Step 1: Clone Repository
```bash
git clone https://github.com/<YOUR_USERNAME>/Truthlens-AI.git
cd Truthlens-AI
```

### Step 2: Configure Environment Variables
```bash
# Backend configuration
copy .env.example .env
```

To enable cloud LLM reasoning (optional), add your key to `.env`:
```ini
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```
*(If unconfigured, TruthLens runs its built-in offline deterministic reasoning engine).*

### Step 3: Start FastAPI Backend
```bash
# Install Python dependencies (CPU-optimized PyTorch)
pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Launch FastAPI server
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
The backend will be live at `http://127.0.0.1:8000` (Docs at `http://127.0.0.1:8000/docs`).

### Step 4: Start React Frontend
In a new terminal window:
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173` in your browser.

---

## 7. API Reference

### Health Check
```http
GET /health
```
**Response:**
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

### Full Verification Pipeline
```http
POST /api/verify
Content-Type: application/json
```
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
  "final_verdict": "MOSTLY TRUE",
  "truth_confidence": 0.583,
  "summary_reasoning": "Found relevant corroborating documentation from wikipedia.org: \"The James Webb Space Telescope (JWST) is a space telescope...\"",
  "article_classification": {
    "model_name": "DistilBERT (Fine-Tuned ISOT, max_len=256)",
    "predicted_label": "REAL",
    "confidence": 0.9998,
    "probabilities": { "REAL": 0.9998, "FAKE": 0.0002 }
  },
  "extracted_claims": [
    {
      "id": 1,
      "claim_text": "NASA's James Webb Space Telescope detected atmospheric water vapor on exoplanet WASP-96b.",
      "verdict": "MOSTLY TRUE",
      "confidence": 0.583,
      "evidence_count": 1
    }
  ],
  "supporting_evidence": [
    {
      "id": 1,
      "title": "Wikipedia: James Webb Space Telescope",
      "snippet": "The James Webb Space Telescope (JWST) is a space telescope designed to conduct infrared astronomy...",
      "source_url": "https://en.wikipedia.org/wiki/James_Webb_Space_Telescope",
      "domain": "wikipedia.org",
      "stance": "SUPPORTING",
      "relevance_score": 0.58,
      "credibility_tier": "Reputable Reference"
    }
  ],
  "contradicting_evidence": [],
  "source_credibility": {
    "overall_rating": "High Reliability",
    "corroboration_score": 100.0
  }
}
```

---

## 8. Deployment Architecture

### Frontend (Vercel)
The React single-page application is deployed via Vercel Edge Network:
1. Connect GitHub repository to **Vercel**.
2. Set **Root Directory** to `frontend`.
3. Add Environment Variable:
   ```env
   VITE_API_BASE_URL=https://truthlens-ai-wag8.onrender.com
   ```
4. Deploy!

### Backend (Render Web Service)
The FastAPI + PyTorch backend is deployed as a managed web service on Render:
1. Connect GitHub repository to **Render** as a **Web Service**.
2. **Build Command**:
   ```bash
   pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt
   ```
3. **Start Command**:
   ```bash
   uvicorn backend.main:app --host 0.0.0.0 --port $PORT
   ```
4. **Environment Variables**:
   ```env
   PORT=10000
   CORS_ORIGINS=*
   LLM_PROVIDER=offline
   ```

---

## 9. Technology Stack

* **Machine Learning & NLP**: PyTorch, Hugging Face Transformers (`distilbert-base-uncased`), XGBoost, Scikit-learn, Joblib, NumPy, Pandas.
* **Backend Framework**: FastAPI, Uvicorn, Pydantic v2, HTTPX, Python-dotenv.
* **Retrieval & Evidence**: Google News RSS, Wikipedia REST API, DuckDuckGo HTML Search.
* **Frontend**: React 19, Vite, Lucide React Icons.
* **Deployment & Cloud**: Vercel (Frontend), Render (FastAPI Web Service), Docker.

---

## 10. License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
