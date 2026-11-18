import { useState } from "react";
import {
  Search,
  ShieldCheck,
  FileText,
  ArrowRight,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  HelpCircle,
  ExternalLink,
  RotateCcw,
  Sparkles,
  Layers,
  Cpu,
  Globe,
  BookOpen,
  Info,
  Clock,
  Check,
  AlertCircle
} from "lucide-react";
import "./App.css";

const SAMPLE_INPUTS = [
  {
    label: "James Webb Exoplanet",
    category: "Science Fact",
    text: "NASA's James Webb Space Telescope detected atmospheric water vapor on exoplanet WASP-96b.",
  },
  {
    label: "Pope Endorsement Hoax",
    category: "Viral Disinformation",
    text: "Pope Francis shocked the world and endorsed Donald Trump for president in an official Vatican statement.",
  },
  {
    label: "Education Scholarships",
    category: "Policy Claim",
    text: "The government announced a new education policy. It will provide scholarships to students from low-income families.",
  },
  {
    label: "Unverified Prediction",
    category: "Speculation",
    text: "Quantum cryptocurrency will completely replace fiat paper money in all world banks by next year.",
  },
];

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

function App() {
  const [claim, setClaim] = useState("");
  const [classifierModel, setClassifierModel] = useState("distilbert");
  const [showResult, setShowResult] = useState(false);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [loadingStep, setLoadingStep] = useState("");
  const [errorMessage, setErrorMessage] = useState("");

  const handleVerify = async () => {
    if (!claim.trim()) {
      setErrorMessage("Please enter a news headline, article, or claim to verify.");
      return;
    }

    setErrorMessage("");
    setLoading(true);
    setLoadingStep("Extracting factual claims & running article classification...");

    try {
      const response = await fetch(`${API_BASE_URL}/api/verify`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          text: claim,
          classifier_model: classifierModel,
          include_evidence: true,
          llm_reasoning: true,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server returned error status ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
      setShowResult(true);
    } catch (error) {
      console.error("Verification failed:", error);
      setErrorMessage(
        error.message.includes("Failed to fetch")
          ? `Unable to connect to TruthLens backend at ${API_BASE_URL}. Ensure the FastAPI server is running.`
          : error.message
      );
    } finally {
      setLoading(false);
      setLoadingStep("");
    }
  };

  const handleReset = () => {
    setClaim("");
    setResult(null);
    setShowResult(false);
    setErrorMessage("");
  };

  const handleSelectSample = (sampleText) => {
    setClaim(sampleText);
    setErrorMessage("");
  };

  const getVerdictTheme = (verdict) => {
    switch (verdict) {
      case "TRUE":
      case "MOSTLY TRUE":
        return {
          colorClass: "verdict-true",
          icon: <CheckCircle2 size={32} />,
          badge: "VERIFIED FACT",
        };
      case "FALSE":
      case "MOSTLY FALSE":
        return {
          colorClass: "verdict-false",
          icon: <XCircle size={32} />,
          badge: "REFUTED / FALSE",
        };
      case "PARTIALLY TRUE":
        return {
          colorClass: "verdict-partial",
          icon: <AlertTriangle size={32} />,
          badge: "PARTIALLY TRUE",
        };
      case "INSUFFICIENT EVIDENCE":
      case "UNVERIFIED":
      default:
        return {
          colorClass: "verdict-unverified",
          icon: <HelpCircle size={32} />,
          badge: "UNVERIFIED / INSUFFICIENT EVIDENCE",
        };
    }
  };

  return (
    <div className="app">
      {/* ================= NAVBAR ================= */}
      <nav className="navbar">
        <div className="brand" onClick={handleReset} style={{ cursor: "pointer" }}>
          <div className="brand-icon">
            <ShieldCheck size={22} />
          </div>
          <div className="brand-text">
            <span>TruthLens AI</span>
            <span className="brand-subtitle">Fact Verification & RAG Engine</span>
          </div>
        </div>

        <div className="nav-links">
          <a href="#home">Home</a>
          <a href="#how-it-works">Architecture</a>
          <a href="#models">Models</a>
          <a href="#about">About</a>
        </div>
      </nav>

      {/* ================= MAIN CONTAINER ================= */}
      <main>
        {/* ================= HERO / INPUT SECTION ================= */}
        {!showResult && (
          <section id="home" className="hero-section">
            <div className="hero-badge">
              <Sparkles size={16} />
              <span>Multi-Stage Fake News Detection & Evidence-Based Verification</span>
            </div>

            <h1>
              Investigate Claims.
              <br />
              <span>Discover the Evidence.</span>
            </h1>

            <p className="hero-description">
              TruthLens AI doesn&apos;t just predict whether news looks fake—it extracts factual
              assertions, retrieves corroborating and contradicting evidence from authoritative sources,
              evaluates source credibility, and explains why.
            </p>

            {/* ERROR BANNER */}
            {errorMessage && (
              <div className="error-banner">
                <AlertCircle size={20} />
                <span>{errorMessage}</span>
                <button onClick={() => setErrorMessage("")} className="close-error">
                  ×
                </button>
              </div>
            )}

            {/* VERIFICATION CARD */}
            <div className="verification-card">
              <div className="input-toolbar">
                <div className="input-header">
                  <FileText size={18} />
                  <span>Submit a news article, headline, or claim</span>
                </div>

                {/* CLASSIFIER MODEL SELECTOR */}
                <div className="model-selector">
                  <Cpu size={15} />
                  <label htmlFor="model-select">Classifier:</label>
                  <select
                    id="model-select"
                    value={classifierModel}
                    onChange={(e) => setClassifierModel(e.target.value)}
                    disabled={loading}
                  >
                    <option value="distilbert">DistilBERT Transformer (ISOT)</option>
                    <option value="xgboost">TF-IDF + XGBoost Baseline</option>
                  </select>
                </div>
              </div>

              <textarea
                value={claim}
                onChange={(e) => setClaim(e.target.value)}
                placeholder="Paste an article, headline, political speech excerpt, or single claim here (e.g. 'NASA detected water vapor on exoplanet WASP-96b')..."
                rows={7}
                disabled={loading}
              />

              {/* SAMPLES ROW */}
              <div className="sample-inputs-wrapper">
                <span className="sample-label">Try sample:</span>
                <div className="sample-chips">
                  {SAMPLE_INPUTS.map((sample, idx) => (
                    <button
                      key={idx}
                      type="button"
                      className="sample-chip"
                      onClick={() => handleSelectSample(sample.text)}
                      disabled={loading}
                    >
                      <span className="chip-cat">{sample.category}:</span> {sample.label}
                    </button>
                  ))}
                </div>
              </div>

              <div className="input-footer">
                <span className="char-count">{claim.length} characters</span>

                <button
                  className="verify-button"
                  onClick={handleVerify}
                  disabled={loading || !claim.trim()}
                >
                  <Search size={18} />
                  <span>{loading ? "Investigating..." : "Verify with TruthLens"}</span>
                  <ArrowRight size={18} />
                </button>
              </div>

              {/* LOADING INDICATOR */}
              {loading && (
                <div className="loading-state">
                  <div className="spinner"></div>
                  <div className="loading-text">
                    <p className="loading-title">TruthLens Pipeline Executing</p>
                    <p className="loading-subtitle">{loadingStep}</p>
                  </div>
                </div>
              )}
            </div>
          </section>
        )}

        {/* ================= VERIFICATION DASHBOARD ================= */}
        {showResult && result && (
          <section className="result-section">
            <div className="result-top">
              <div>
                <div className="result-meta-row">
                  <span className="result-badge">EVIDENCE-GROUNDED VERIFICATION REPORT</span>
                  {result.execution_time_seconds > 0 && (
                    <span className="time-badge">
                      <Clock size={13} /> {result.execution_time_seconds}s
                    </span>
                  )}
                </div>
                <h1>Verification Analysis</h1>
              </div>

              <button className="reset-button" onClick={handleReset}>
                <RotateCcw size={17} />
                Investigate another claim
              </button>
            </div>

            {/* ================= PIPELINE STEP TAGS ================= */}
            {result.pipeline_steps && result.pipeline_steps.length > 0 && (
              <div className="pipeline-steps-bar">
                <span className="pipeline-label">Pipeline Trace:</span>
                {result.pipeline_steps.map((step, idx) => (
                  <span key={idx} className="pipeline-tag">
                    <Check size={12} /> {step}
                  </span>
                ))}
              </div>
            )}

            {/* ================= 1. OVERALL FACT VERDICT BANNER ================= */}
            {(() => {
              const theme = getVerdictTheme(result.final_verdict);
              return (
                <div className={`verdict-card ${theme.colorClass}`}>
                  <div className="verdict-icon">{theme.icon}</div>

                  <div className="verdict-content">
                    <div className="verdict-badge-row">
                      <span className="verdict-status">{result.final_verdict}</span>
                      <span className="verdict-type-pill">FACT VERIFICATION</span>
                    </div>

                    <h2>{result.summary_reasoning}</h2>

                    <p className="verdict-notice">
                      Verdict synthesized from multi-source retrieved evidence passages and
                      corroboration analysis.
                    </p>
                  </div>

                  <div className="confidence-box">
                    <span>Truth Confidence</span>
                    <strong>{(result.confidence_score * 100).toFixed(1)}%</strong>
                    <div className="mini-meter">
                      <div
                        className="mini-meter-fill"
                        style={{ width: `${result.confidence_score * 100}%` }}
                      ></div>
                    </div>
                  </div>
                </div>
              );
            })()}

            {/* ================= 2. TWO-COLUMN ARCHITECTURE DISTINCTION ================= */}
            <div className="two-column-grid">
              {/* CARD A: ARTICLE CLASSIFICATION (ISOT ML MODEL) */}
              <div className="dashboard-card model-distinction-card">
                <div className="card-header">
                  <div className="header-icon-title">
                    <Cpu size={19} className="accent-blue" />
                    <span>1. Article Classification (Stylistic Pattern)</span>
                  </div>
                  <span className="subtle-pill">
                    {result.article_classification?.model_name || "ML Classifier"}
                  </span>
                </div>

                <div className="classification-box">
                  <div className="classification-header">
                    <span>Stylistic Prediction:</span>
                    <strong
                      className={
                        result.article_classification?.predicted_label === "REAL"
                          ? "status-real"
                          : "status-fake"
                      }
                    >
                      {result.article_classification?.predicted_label === "REAL"
                        ? "REAL NEWS STYLE"
                        : "FAKE NEWS STYLE"}
                    </strong>
                  </div>

                  <div className="probability-bars">
                    <div className="prob-row">
                      <span>Real News Style</span>
                      <span>
                        {(
                          (result.article_classification?.probabilities?.REAL ??
                            result.confidence) * 100
                        ).toFixed(1)}
                        %
                      </span>
                    </div>
                    <div className="bar-track">
                      <div
                        className="bar-fill real"
                        style={{
                          width: `${(result.article_classification?.probabilities?.REAL ?? result.confidence) * 100}%`,
                        }}
                      ></div>
                    </div>

                    <div className="prob-row" style={{ marginTop: "8px" }}>
                      <span>Disinformation / Fake Style</span>
                      <span>
                        {(
                          (result.article_classification?.probabilities?.FAKE ??
                            1 - result.confidence) * 100
                        ).toFixed(1)}
                        %
                      </span>
                    </div>
                    <div className="bar-track">
                      <div
                        className="bar-fill fake"
                        style={{
                          width: `${(result.article_classification?.probabilities?.FAKE ?? 1 - result.confidence) * 100}%`,
                        }}
                      ></div>
                    </div>
                  </div>

                  <div className="model-explanation-callout">
                    <Info size={16} />
                    <p>
                      <strong>Important Architectural Distinction:</strong> This model analyzes
                      journalistic syntax and stylistic n-grams learned from the ISOT dataset. It does
                      not search the live internet or confirm real-world empirical ground truth on its
                      own.
                    </p>
                  </div>
                </div>
              </div>

              {/* CARD B: SOURCE CREDIBILITY & CORROBORATION */}
              <div className="dashboard-card">
                <div className="card-header">
                  <div className="header-icon-title">
                    <Globe size={19} className="accent-blue" />
                    <span>2. Source Credibility & Corroboration</span>
                  </div>
                  <span className="subtle-pill">Domain Verification</span>
                </div>

                <div className="source-analysis-content">
                  <div className="source-rating-header">
                    <div>
                      <span className="label-sm">Overall Reliability Tier</span>
                      <p className="rating-text">
                        {result.source_credibility?.overall_rating || "Insufficient information"}
                      </p>
                    </div>

                    <div className="corroboration-stat">
                      <span className="label-sm">Corroboration</span>
                      <strong>
                        {((result.source_credibility?.corroboration_score || 0) * 100).toFixed(0)}%
                      </strong>
                    </div>
                  </div>

                  <div className="corroboration-meter">
                    <div
                      className="corroboration-fill"
                      style={{
                        width: `${(result.source_credibility?.corroboration_score || 0) * 100}%`,
                      }}
                    ></div>
                  </div>

                  <p className="source-notes">{result.source_credibility?.notes}</p>

                  {/* DOMAIN CHIPS */}
                  <div className="domain-chips-list">
                    <span className="label-sm">Evaluated Domains:</span>
                    {result.source_credibility?.evaluated_sources &&
                    result.source_credibility.evaluated_sources.length > 0 ? (
                      <div className="domain-chips">
                        {result.source_credibility.evaluated_sources.map((src, i) => (
                          <div key={i} className="domain-chip">
                            <span className="domain-name">{src.domain}</span>
                            <span className="domain-tier">{src.credibility_tier}</span>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <p className="empty-subtle">
                        No external domains were retrieved to establish credibility.
                      </p>
                    )}
                  </div>
                </div>
              </div>
            </div>

            {/* ================= 3. EXTRACTED FACTUAL CLAIMS ================= */}
            <div className="dashboard-card">
              <div className="card-header">
                <div className="header-icon-title">
                  <Layers size={19} className="accent-blue" />
                  <span>3. Extracted Factual Claims</span>
                </div>
                <span className="subtle-pill">
                  {result.extracted_claims?.length || 0} Claim(s) Investigated
                </span>
              </div>

              <div className="claims-list">
                {result.extracted_claims && result.extracted_claims.length > 0 ? (
                  result.extracted_claims.map((claimItem) => {
                    const claimTheme = getVerdictTheme(claimItem.verdict);
                    return (
                      <div key={claimItem.id} className="claim-card">
                        <div className="claim-header">
                          <span className="claim-id">Claim #{claimItem.id}</span>
                          <span className={`claim-verdict-badge ${claimTheme.colorClass}`}>
                            {claimItem.verdict}
                          </span>
                        </div>

                        <p className="claim-text">&quot;{claimItem.claim_text}&quot;</p>

                        {claimItem.reasoning && (
                          <p className="claim-reasoning">
                            <strong>Reasoning:</strong> {claimItem.reasoning}
                          </p>
                        )}
                      </div>
                    );
                  })
                ) : (
                  <p className="empty-subtle">
                    No discrete factual claims were separated; analyzed as a single proposition.
                  </p>
                )}
              </div>
            </div>

            {/* ================= 4. EVIDENCE DOSSIER ================= */}
            <div className="two-column-grid">
              {/* SUPPORTING EVIDENCE */}
              <div className="dashboard-card">
                <div className="card-header">
                  <div className="header-icon-title">
                    <CheckCircle2 size={19} className="accent-green" />
                    <span>Supporting Evidence</span>
                  </div>
                  <span className="subtle-pill">
                    {result.supporting_evidence?.length || 0} Source(s)
                  </span>
                </div>

                <div className="evidence-items-container">
                  {result.supporting_evidence && result.supporting_evidence.length > 0 ? (
                    result.supporting_evidence.map((item, idx) => (
                      <div key={idx} className="evidence-card supporting">
                        <div className="evidence-card-top">
                          <span className="evidence-domain">{item.domain || "Authoritative Source"}</span>
                          <span className="relevance-tag">
                            Relevance: {(item.relevance_score * 100).toFixed(0)}%
                          </span>
                        </div>

                        <h4 className="evidence-title">{item.title}</h4>
                        <p className="evidence-snippet">&quot;{item.snippet}&quot;</p>

                        {item.source_url && (
                          <a
                            href={item.source_url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="evidence-link"
                          >
                            <span>Inspect Source Document</span>
                            <ExternalLink size={14} />
                          </a>
                        )}
                      </div>
                    ))
                  ) : (
                    <p className="empty-subtle">
                      No explicit corroborating evidence passages identified in indexed sources.
                    </p>
                  )}
                </div>
              </div>

              {/* CONTRADICTING EVIDENCE */}
              <div className="dashboard-card">
                <div className="card-header">
                  <div className="header-icon-title">
                    <AlertTriangle size={19} className="accent-red" />
                    <span>Contradicting / Refuting Evidence</span>
                  </div>
                  <span className="subtle-pill">
                    {result.contradicting_evidence?.length || 0} Source(s)
                  </span>
                </div>

                <div className="evidence-items-container">
                  {result.contradicting_evidence && result.contradicting_evidence.length > 0 ? (
                    result.contradicting_evidence.map((item, idx) => (
                      <div key={idx} className="evidence-card contradicting">
                        <div className="evidence-card-top">
                          <span className="evidence-domain">{item.domain || "Fact Checker / Debunk"}</span>
                          <span className="relevance-tag refutation">
                            Refutation Signal
                          </span>
                        </div>

                        <h4 className="evidence-title">{item.title}</h4>
                        <p className="evidence-snippet">&quot;{item.snippet}&quot;</p>

                        {item.source_url && (
                          <a
                            href={item.source_url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="evidence-link"
                          >
                            <span>Inspect Counter-Evidence</span>
                            <ExternalLink size={14} />
                          </a>
                        )}
                      </div>
                    ))
                  ) : (
                    <p className="empty-subtle">
                      No direct counter-evidence or debunking records detected for this statement.
                    </p>
                  )}
                </div>
              </div>
            </div>

            {/* ================= 5. RESEARCH DISCLAIMER ================= */}
            <div className="disclaimer-footer-card">
              <BookOpen size={18} />
              <p>{result.disclaimer}</p>
            </div>
          </section>
        )}

        {/* ================= HOW IT WORKS SECTION ================= */}
        <section id="how-it-works" className="how-section">
          <div className="section-heading">
            <p>ARCHITECTURE & METHODOLOGY</p>
            <h2>How TruthLens AI Investigates</h2>
            <span>
              Moving beyond simplistic binary classification into evidence-backed factual verification.
            </span>
          </div>

          <div className="feature-grid">
            <div className="feature-card">
              <div className="feature-number">01</div>
              <h3>Dual-Layer Architecture</h3>
              <p>
                Clearly separates <strong>Article Stylistic Classification</strong> (DistilBERT
                transformer fine-tuned on ISOT, and classical XGBoost) from genuine{" "}
                <strong>Fact Verification</strong>.
              </p>
            </div>

            <div className="feature-card">
              <div className="feature-number">02</div>
              <h3>Claim Extraction</h3>
              <p>
                Decomposes submitted text into discrete verifiable factual propositions, filtering out
                subjective opinion and rhetorical hyperbole.
              </p>
            </div>

            <div className="feature-card">
              <div className="feature-number">03</div>
              <h3>RAG & Multi-Source Retrieval</h3>
              <p>
                Retrieves real-world evidence from encyclopedias, government portals, and reputable wire
                services, ranking passages by semantic relevance and stance.
              </p>
            </div>

            <div className="feature-card">
              <div className="feature-number">04</div>
              <h3>Evidence-Grounded Reasoning</h3>
              <p>
                Synthesizes verdicts (TRUE, MOSTLY TRUE, PARTIALLY TRUE, FALSE, UNVERIFIED) with prompt
                injection defenses and strict non-hallucination rules.
              </p>
            </div>

            <div className="feature-card">
              <div className="feature-number">05</div>
              <h3>Source Credibility Analysis</h3>
              <p>
                Assesses institutional standing, domain history, and multi-source corroboration without
                inventing arbitrary fake confidence numbers.
              </p>
            </div>

            <div className="feature-card">
              <div className="feature-number">06</div>
              <h3>Explainable Dashboard</h3>
              <p>
                Every conclusion provides clickable source citations, transparent probability
                breakdowns, and execution telemetry for research reproducibility.
              </p>
            </div>
          </div>
        </section>

        {/* ================= MODELS COMPARISON SECTION ================= */}
        <section id="models" className="models-section">
          <div className="section-heading">
            <p>BENCHMARKS</p>
            <h2>Empirical Model Performance</h2>
            <span>
              Rigorously evaluated on the held-out test split of the 39,103 article ISOT Dataset.
            </span>
          </div>

          <div className="table-wrapper">
            <table className="benchmark-table">
              <thead>
                <tr>
                  <th>Model Architecture</th>
                  <th>Features / Input</th>
                  <th>Accuracy</th>
                  <th>Precision</th>
                  <th>Recall</th>
                  <th>F1 Score</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>
                    <strong>DistilBERT</strong> (Fine-Tuned Checkpoint-7821)
                  </td>
                  <td>Transformer Embeddings (256 Tokens)</td>
                  <td>
                    <span className="badge-highlight">99.87%</span>
                  </td>
                  <td>99.91%</td>
                  <td>99.86%</td>
                  <td>
                    <span className="badge-highlight">99.88%</span>
                  </td>
                </tr>
                <tr>
                  <td>
                    <strong>TF-IDF + XGBoost</strong> (300 Hist Trees)
                  </td>
                  <td>50,000 N-Grams (1,2)</td>
                  <td>
                    <span className="badge-highlight">99.65%</span>
                  </td>
                  <td>99.51%</td>
                  <td>99.86%</td>
                  <td>
                    <span className="badge-highlight">99.68%</span>
                  </td>
                </tr>
                <tr>
                  <td>
                    <strong>TF-IDF + Logistic Regression</strong>
                  </td>
                  <td>50,000 N-Grams (Title + Text)</td>
                  <td>98.61%</td>
                  <td>98.11%</td>
                  <td>99.34%</td>
                  <td>98.72%</td>
                </tr>
                <tr>
                  <td>
                    <strong>TF-IDF + Naive Bayes</strong>
                  </td>
                  <td>50,000 N-Grams (Title + Text)</td>
                  <td>95.69%</td>
                  <td>96.01%</td>
                  <td>96.04%</td>
                  <td>96.03%</td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>
      </main>

      {/* ================= FOOTER ================= */}
      <footer id="about">
        <div className="brand">
          <div className="brand-icon">
            <ShieldCheck size={20} />
          </div>
          <span>TruthLens AI</span>
        </div>

        <p className="footer-credits">
          AI-Powered News & Evidence-Based Fact Verification • Research Pipeline v2.0
        </p>
      </footer>
    </div>
  );
}

export default App;
