from pydantic import BaseModel, Field
from typing import Optional


class VerificationRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=3,
        max_length=50000,
        description="News headline, article, or factual claim to verify."
    )
    include_evidence: bool = Field(
        default=True,
        description="Whether to perform claim extraction and evidence retrieval."
    )
    llm_reasoning: bool = Field(
        default=True,
        description="Whether to use the LLM reasoning pipeline over retrieved evidence."
    )
    classifier_model: str = Field(
        default="distilbert",
        description="Article classifier to use: 'distilbert', 'xgboost', or 'both'."
    )


class ClassificationRequest(BaseModel):
    text: str = Field(..., min_length=3, max_length=50000)
    model: str = Field(default="distilbert", description="'distilbert' or 'xgboost'")


class ClaimExtractionRequest(BaseModel):
    text: str = Field(..., min_length=3, max_length=50000)


class EvidenceRetrievalRequest(BaseModel):
    claim: str = Field(..., min_length=3, max_length=1000)
    max_results: int = Field(default=5, ge=1, le=15)
