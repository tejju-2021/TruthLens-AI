from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class ArticleClassificationResult(BaseModel):
    model_name: str
    predicted_label: str = Field(..., description="'REAL' or 'FAKE'")
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: Dict[str, float] = Field(default_factory=dict)
    explanation: str = Field(
        default="Classification based on ISOT news dataset linguistic style and patterns."
    )


class EvidenceItem(BaseModel):
    id: int
    title: str
    snippet: str
    source_url: Optional[str] = None
    domain: Optional[str] = None
    stance: str = Field(
        default="NEUTRAL",
        description="'SUPPORTING', 'CONTRADICTING', 'NEUTRAL', or 'INCONCLUSIVE'"
    )
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    published_date: Optional[str] = None
    credibility_tier: Optional[str] = Field(
        default="Unknown",
        description="'Official/Institutional', 'Reputable News', 'General Web', 'Low Reliability'"
    )


class ClaimItem(BaseModel):
    id: int
    claim_text: str
    is_verifiable: bool = True
    verdict: str = Field(
        default="UNVERIFIED",
        description="'TRUE', 'MOSTLY TRUE', 'PARTIALLY TRUE', 'MOSTLY FALSE', 'FALSE', 'UNVERIFIED', 'INSUFFICIENT EVIDENCE'"
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_count: int = 0
    supporting_evidence: List[EvidenceItem] = Field(default_factory=list)
    contradicting_evidence: List[EvidenceItem] = Field(default_factory=list)
    reasoning: Optional[str] = None


class SourceEvaluationItem(BaseModel):
    domain: str
    source_type: str
    credibility_tier: str
    confidence_assessment: str
    is_official_or_academic: bool = False


class SourceCredibilityResult(BaseModel):
    overall_rating: str = Field(
        default="Insufficient information",
        description="'High Reliability', 'Moderate Reliability', 'Mixed/Caution', 'Low Reliability', 'Insufficient information'"
    )
    evaluated_sources: List[SourceEvaluationItem] = Field(default_factory=list)
    corroboration_score: float = Field(default=0.0, ge=0.0, le=1.0)
    notes: str = ""


class VerificationResponse(BaseModel):
    # Top-level legacy fields for backwards compatibility with existing UI
    verdict: str = Field(..., description="Legacy verdict field: 'REAL' or 'FAKE'")
    confidence: float = Field(..., description="Legacy confidence score (0.0 to 1.0)")
    
    # Advanced research & verification fields
    query: str
    final_verdict: str = Field(
        ...,
        description="'TRUE', 'MOSTLY TRUE', 'PARTIALLY TRUE', 'MOSTLY FALSE', 'FALSE', 'UNVERIFIED', 'INSUFFICIENT EVIDENCE'"
    )
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    summary_reasoning: str
    article_classification: ArticleClassificationResult
    extracted_claims: List[ClaimItem] = Field(default_factory=list)
    supporting_evidence: List[EvidenceItem] = Field(default_factory=list)
    contradicting_evidence: List[EvidenceItem] = Field(default_factory=list)
    source_credibility: SourceCredibilityResult
    pipeline_steps: List[str] = Field(default_factory=list)
    execution_time_seconds: float = 0.0
    disclaimer: str = (
        "TruthLens AI provides evidence-grounded analysis. Always verify critical facts with primary authoritative sources."
    )


class HealthResponse(BaseModel):
    status: str
    project: str
    version: str
    distilbert_loaded: bool
    xgboost_available: bool
    llm_configured: bool
    llm_provider: str
    retrieval_provider: str
