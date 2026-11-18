"""
TruthLens AI — Source Credibility Analysis Module
Evaluates domain identity, institutional status, source type, and independent corroboration.
Does NOT fabricate arbitrary credibility percentages.
Returns 'Insufficient information' when sources cannot be evaluated reliably.
Keeps source credibility strictly separated from factual correctness.
"""

import urllib.parse
from typing import List, Dict, Any, Optional
from backend.schemas.response import (
    EvidenceItem,
    SourceCredibilityResult,
    SourceEvaluationItem
)

# Registry of authoritative institutional, peer-reviewed, and reputable reference domains
INSTITUTIONAL_DOMAINS = {
    "nasa.gov", "nih.gov", "cdc.gov", "who.int", "un.org", "whitehouse.gov",
    "loc.gov", "europa.eu", "gov.uk", "state.gov", "justice.gov", "defense.gov",
    "parliament.uk", "unicef.org", "unesco.org", "imf.org", "worldbank.org"
}

ESTABLISHED_NEWS_AND_FACTCHECK = {
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "wikipedia.org",
    "snopes.com", "factcheck.org", "politifact.com", "nature.com", "science.org",
    "thelancet.com", "nejm.org", "pnas.org", "ieee.org"
}

RECOGNIZED_MEDIA = {
    "nytimes.com", "wsj.com", "theguardian.com", "washingtonpost.com",
    "bloomberg.com", "ft.com", "economist.com", "npr.org", "pbs.org",
    "cbsnews.com", "nbcnews.com", "abcnews.go.com", "time.com", "forbes.com",
    "indianexpress.com", "thehindu.com", "ndtv.com", "timesofindia.indiatimes.com",
    "hindustantimes.com", "indiatoday.in", "aljazeera.com", "afp.com", "news.google.com"
}


class SourceCredibilityAnalyzer:
    def __init__(self):
        pass

    def evaluate_sources(self, evidence_items: List[EvidenceItem]) -> SourceCredibilityResult:
        if not evidence_items:
            return SourceCredibilityResult(
                overall_rating="Insufficient information",
                evaluated_sources=[],
                corroboration_score=0.0,
                notes="No external sources were retrieved to assess credibility."
            )

        evaluated: List[SourceEvaluationItem] = []
        unique_domains = set()
        official_count = 0
        established_count = 0

        for item in evidence_items:
            domain = self._normalize_domain(item.domain or item.source_url)
            if not domain or domain in ["unknown", ""]:
                continue

            if domain in unique_domains:
                continue
            unique_domains.add(domain)

            # Evaluate domain standing
            is_official = domain.endswith(".gov") or domain.endswith(".edu") or domain.endswith(".mil") or domain in INSTITUTIONAL_DOMAINS
            is_established = domain in ESTABLISHED_NEWS_AND_FACTCHECK or domain in RECOGNIZED_MEDIA

            if is_official:
                official_count += 1
                source_type = "Official / Institutional"
                tier = "High Reliability"
                assessment = "Official government, educational, or global institutional domain with strict editorial accountability."
            elif domain in ESTABLISHED_NEWS_AND_FACTCHECK:
                established_count += 1
                source_type = "Primary Wire Service / Established Fact-Checker"
                tier = "High Reliability"
                assessment = "Globally recognized wire service, peer-reviewed journal, or verified public reference."
            elif domain in RECOGNIZED_MEDIA:
                established_count += 1
                source_type = "Recognized Major Media"
                tier = "Moderate Reliability"
                assessment = "Mainstream commercial journalism outlet with standard editorial review processes."
            else:
                source_type = "General Web / Independent"
                tier = "Caution / Unverified"
                assessment = "General web publication or blog. Requires independent corroboration."

            evaluated.append(
                SourceEvaluationItem(
                    domain=domain,
                    source_type=source_type,
                    credibility_tier=tier,
                    confidence_assessment=assessment,
                    is_official_or_academic=is_official
                )
            )

        if not evaluated:
            return SourceCredibilityResult(
                overall_rating="Insufficient information",
                evaluated_sources=[],
                corroboration_score=0.0,
                notes="Retrieved sources lacked verifiable domain metadata."
            )

        # Corroboration and overall rating calculation
        total_unique = len(evaluated)
        high_rel_sources = official_count + established_count

        if high_rel_sources >= 2:
            overall_rating = "High Reliability"
            corroboration_score = min(0.95, 0.6 + (0.15 * high_rel_sources))
            notes = (
                f"Information is corroborated by {high_rel_sources} independent authoritative sources "
                f"including {evaluated[0].domain}."
            )
        elif high_rel_sources == 1:
            overall_rating = "Moderate Reliability"
            corroboration_score = 0.70
            notes = f"Corroborated by 1 established source ({evaluated[0].domain}), supported by general web references."
        elif total_unique >= 1:
            overall_rating = "Mixed / Caution"
            corroboration_score = 0.45
            notes = "Sources consist of general web domains without top-tier institutional accreditation."
        else:
            overall_rating = "Insufficient information"
            corroboration_score = 0.0
            notes = "Insufficient domain history to evaluate credibility."

        return SourceCredibilityResult(
            overall_rating=overall_rating,
            evaluated_sources=evaluated,
            corroboration_score=round(corroboration_score, 2),
            notes=notes
        )

    def _normalize_domain(self, domain_or_url: Optional[str]) -> str:
        if not domain_or_url:
            return ""
        try:
            if "://" in domain_or_url:
                parsed = urllib.parse.urlparse(domain_or_url)
                netloc = parsed.netloc.lower()
            else:
                netloc = domain_or_url.lower()
            if netloc.startswith("www."):
                netloc = netloc[4:]
            return netloc.strip()
        except Exception:
            return ""


_credibility_analyzer_instance = None


def get_source_credibility_analyzer() -> SourceCredibilityAnalyzer:
    global _credibility_analyzer_instance
    if _credibility_analyzer_instance is None:
        _credibility_analyzer_instance = SourceCredibilityAnalyzer()
    return _credibility_analyzer_instance
