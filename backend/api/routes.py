import time
import logging
from fastapi import APIRouter, HTTPException
from backend.config import settings
from backend.schemas.request import (
    VerificationRequest,
    ClassificationRequest,
    ClaimExtractionRequest,
    EvidenceRetrievalRequest
)
from backend.schemas.response import (
    VerificationResponse,
    ArticleClassificationResult,
    HealthResponse,
    ClaimItem,
    EvidenceItem,
    SourceCredibilityResult
)
from backend.services.classifier import DistilBERTClassifier, XGBoostClassifier
from backend.services.claim_extractor import get_claim_extractor
from backend.services.evidence_retriever import get_evidence_retriever
from backend.services.evidence_ranker import get_evidence_ranker
from backend.services.rag_pipeline import get_rag_pipeline
from backend.services.llm_verifier import get_llm_verifier
from backend.services.source_credibility import get_source_credibility_analyzer

logger = logging.getLogger("truthlens.api")
router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """System health check and module readiness status."""
    distilbert_loaded = False
    try:
        classifier = DistilBERTClassifier.get_instance()
        distilbert_loaded = classifier.is_loaded
    except Exception:
        distilbert_loaded = False

    xgboost_available = settings.XGBOOST_MODEL_PATH.exists() and settings.TFIDF_VECTORIZER_PATH.exists()
    llm_configured = bool(settings.GROQ_API_KEY or settings.OPENAI_API_KEY)

    return HealthResponse(
        status="healthy",
        project=settings.PROJECT_NAME,
        version=settings.VERSION,
        distilbert_loaded=distilbert_loaded,
        xgboost_available=xgboost_available,
        llm_configured=llm_configured,
        llm_provider=settings.LLM_PROVIDER,
        retrieval_provider=settings.RETRIEVAL_PROVIDER,
    )


@router.post("/api/classify", response_model=ArticleClassificationResult, tags=["Classification"])
def classify_article(request: ClassificationRequest):
    """
    Classify a news article using fine-tuned DistilBERT or classical XGBoost.
    Separates article stylistic classification from factual truth verification.
    """
    try:
        if request.model.lower() == "xgboost":
            xgb_classifier = XGBoostClassifier.get_instance()
            return xgb_classifier.predict(request.text)
        else:
            classifier = DistilBERTClassifier.get_instance()
            return classifier.predict(request.text)
    except Exception as e:
        logger.error(f"Classification error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")


@router.post("/api/extract-claims", tags=["Extraction"])
def extract_claims_endpoint(request: ClaimExtractionRequest):
    """Extract verifiable factual claims from text."""
    try:
        extractor = get_claim_extractor()
        return extractor.extract_claims(request.text)
    except Exception as e:
        logger.error(f"Claim extraction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Claim extraction failed: {str(e)}")


@router.post("/api/retrieve-evidence", tags=["Retrieval"])
def retrieve_evidence_endpoint(request: EvidenceRetrievalRequest):
    """Retrieve and rank external evidence passages for a claim."""
    try:
        retriever = get_evidence_retriever()
        ranker = get_evidence_ranker()
        raw_evidence = retriever.search(request.claim, max_results=request.max_results)
        supporting, contradicting = ranker.rank_evidence(request.claim, raw_evidence)
        return {
            "claim": request.claim,
            "supporting_evidence": supporting,
            "contradicting_evidence": contradicting,
            "total_retrieved": len(raw_evidence)
        }
    except Exception as e:
        logger.error(f"Evidence retrieval error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Evidence retrieval failed: {str(e)}")


@router.post("/api/verify", response_model=VerificationResponse, tags=["Verification"])
@router.post("/verify", response_model=VerificationResponse, tags=["Verification"])
def verify_claim_or_article(request: VerificationRequest):
    """
    Complete TruthLens AI Verification Pipeline:
    1. ISOT Article Classification (DistilBERT / XGBoost)
    2. Claim Extraction (Verifiable propositions)
    3. Multi-Source Evidence Retrieval
    4. Lexical & Stance Evidence Ranking
    5. Context-grounded LLM / Rule Reasoning
    6. Source Credibility Assessment
    7. Synthesized Explainable Verification Report
    """
    start_time = time.time()
    steps_executed = []

    try:
        # Step 1: Article Style Classification (ISOT Model)
        if request.classifier_model.lower() == "xgboost":
            classifier = XGBoostClassifier.get_instance()
        else:
            classifier = DistilBERTClassifier.get_instance()

        cls_result = classifier.predict(request.text)
        steps_executed.append(f"Article Classification ({cls_result.model_name})")

        # Step 2 & 3 & 4: RAG Pipeline (Claims + Retrieval + Ranking)
        rag_pipeline = get_rag_pipeline()
        claims, supporting, contradicting, context_prompt = rag_pipeline.process_text(
            request.text,
            max_claims=3,
            max_evidence_per_claim=4
        )
        steps_executed.append(f"Claim Extraction ({len(claims)} factual claim(s) identified)")
        steps_executed.append(f"Evidence Retrieval ({len(supporting)} supporting, {len(contradicting)} contradicting)")

        # Step 5: LLM Reasoning
        verifier = get_llm_verifier()
        final_verdict, confidence_score, summary_reasoning, updated_claims = verifier.verify(
            original_text=request.text,
            claims=claims,
            supporting=supporting,
            contradicting=contradicting,
            context_prompt=context_prompt
        )
        steps_executed.append("Evidence-Grounded Factual Reasoning")

        # Step 6: Source Credibility Analysis
        cred_analyzer = get_source_credibility_analyzer()
        all_retrieved = supporting + contradicting
        source_cred = cred_analyzer.evaluate_sources(all_retrieved)
        steps_executed.append("Source Credibility Assessment")

        execution_time = time.time() - start_time

        return VerificationResponse(
            verdict=cls_result.predicted_label,
            confidence=cls_result.confidence,
            query=request.text,
            final_verdict=final_verdict,
            confidence_score=round(confidence_score, 4),
            summary_reasoning=summary_reasoning,
            article_classification=cls_result,
            extracted_claims=updated_claims,
            supporting_evidence=supporting,
            contradicting_evidence=contradicting,
            source_credibility=source_cred,
            pipeline_steps=steps_executed,
            execution_time_seconds=round(execution_time, 4),
        )

    except Exception as e:
        logger.error(f"Verification pipeline failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")
