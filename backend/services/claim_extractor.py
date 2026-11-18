"""
TruthLens AI — Claim Extraction Module
Extracts verifiable factual claims from complete articles or individual claims.
Distinguishes verifiable factual statements from opinions and subjective rhetoric.
Preserves original statements without fabricating external claims.
"""

import re
import logging
from typing import List, Optional
from backend.schemas.response import ClaimItem
from backend.config import settings

logger = logging.getLogger("truthlens.claim_extractor")

# Subjective & opinion markers
OPINION_MARKERS = [
    r"\bi (?:think|believe|feel|suppose|guess)\b",
    r"\bin my (?:opinion|view)\b",
    r"\barguably\b",
    r"\bperhaps\b",
    r"\bshould (?:always|never|definitely)\b",
    r"\bit is (?:obvious|clear|certain) that\b",
    r"\bworst (?:ever|in history)\b",
    r"\bgreatest (?:ever|of all time)\b",
    r"\bterrible|horrible|wonderful|magnificent\b",
]

# Verifiable factual indicators (action verbs, quantifiable metrics, declarations)
FACTUAL_VERB_PATTERNS = [
    r"\b(?:announced|approved|signed|voted|passed|enacted|introduced)\b",
    r"\b(?:reported|stated|confirmed|denied|revealed|declared|claimed)\b",
    r"\b(?:increased|decreased|rose|fell|dropped|surpassed|exceeded)\b",
    r"\b(?:arrested|charged|convicted|sentenced|indicted|investigated)\b",
    r"\b(?:launched|built|discovered|created|tested|published|founded)\b",
    r"\b(?:won|lost|defeated|elected|appointed|resigned)\b",
    r"\b(?:killed|injured|died|hospitalized|affected)\b",
]

NUMBER_PATTERN = re.compile(r"\b\d+(?:[.,]\d+)?%?|\b(?:million|billion|trillion|thousand|hundred)\b", re.IGNORECASE)


class ClaimExtractor:
    def __init__(self):
        pass

    def extract_claims(self, text: str, max_claims: int = 5) -> List[ClaimItem]:
        """
        Extract verifiable factual claims from input text.
        If input is a single statement/claim, returns it as claim 1.
        """
        cleaned = text.strip()
        if not cleaned:
            return []

        # Case 1: Short single claim / headline
        sentences = self._split_into_sentences(cleaned)
        if len(sentences) <= 1:
            is_verifiable = self._is_verifiable_statement(cleaned)
            return [
                ClaimItem(
                    id=1,
                    claim_text=cleaned,
                    is_verifiable=is_verifiable,
                    verdict="UNVERIFIED",
                    confidence=0.0,
                    evidence_count=0
                )
            ]

        # Case 2: Multi-sentence article
        claims: List[ClaimItem] = []
        claim_id = 1

        for sentence in sentences:
            sentence_clean = sentence.strip()
            # Ignore tiny fragments
            if len(sentence_clean.split()) < 4:
                continue

            is_verifiable = self._is_verifiable_statement(sentence_clean)
            is_opinion = self._is_predominantly_opinion(sentence_clean)

            # Prioritize verifiable factual assertions over pure opinion
            if is_verifiable and not is_opinion:
                claims.append(
                    ClaimItem(
                        id=claim_id,
                        claim_text=sentence_clean,
                        is_verifiable=True,
                        verdict="UNVERIFIED",
                        confidence=0.0,
                        evidence_count=0
                    )
                )
                claim_id += 1
                if len(claims) >= max_claims:
                    break

        # Fallback: if no sentences passed strict filters, take the first 2-3 longest informative sentences
        if not claims:
            for i, sent in enumerate(sentences[:3]):
                if len(sent.split()) >= 4:
                    claims.append(
                        ClaimItem(
                            id=i + 1,
                            claim_text=sent.strip(),
                            is_verifiable=True,
                            verdict="UNVERIFIED",
                            confidence=0.0,
                            evidence_count=0
                        )
                    )

        return claims

    def _split_into_sentences(self, text: str) -> List[str]:
        """Sentence boundary splitting protecting common abbreviations."""
        # Replace common abbreviations with temp tokens
        subbed = text
        abbrevs = {
            "U.S.": "US_ABBR",
            "U.K.": "UK_ABBR",
            "Dr.": "DR_ABBR",
            "Mr.": "MR_ABBR",
            "Ms.": "MS_ABBR",
            "Sen.": "SEN_ABBR",
            "Rep.": "REP_ABBR",
            "Gov.": "GOV_ABBR",
            "Jan.": "JAN_ABBR",
            "Feb.": "FEB_ABBR",
            "Aug.": "AUG_ABBR",
            "Sept.": "SEPT_ABBR",
            "Oct.": "OCT_ABBR",
            "Nov.": "NOV_ABBR",
            "Dec.": "DEC_ABBR",
            "vs.": "VS_ABBR",
            "e.g.": "EG_ABBR",
            "i.e.": "IE_ABBR",
        }
        for abbr, token in abbrevs.items():
            subbed = subbed.replace(abbr, token)

        # Split on sentence terminals
        raw_sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'“])", subbed)
        
        # Restore tokens
        results = []
        for s in raw_sentences:
            for abbr, token in abbrevs.items():
                s = s.replace(token, abbr)
            s = s.strip()
            if s:
                results.append(s)
        return results

    def _is_verifiable_statement(self, text: str) -> bool:
        """Determines if a statement makes a factual, checkable assertion."""
        # Check for numbers, statistics, or metrics
        has_number = bool(NUMBER_PATTERN.search(text))
        
        # Check for factual action verbs
        has_factual_verb = any(re.search(pat, text, re.IGNORECASE) for pat in FACTUAL_VERB_PATTERNS)
        
        # Check for named entities or capitalized subject tokens
        has_named_subject = bool(re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text))

        return (has_factual_verb and (has_number or has_named_subject)) or (has_number and len(text.split()) > 5)

    def _is_predominantly_opinion(self, text: str) -> bool:
        """Checks if a statement is primarily subjective opinion or rhetorical."""
        for pattern in OPINION_MARKERS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False


_claim_extractor_instance = None


def get_claim_extractor() -> ClaimExtractor:
    global _claim_extractor_instance
    if _claim_extractor_instance is None:
        _claim_extractor_instance = ClaimExtractor()
    return _claim_extractor_instance
