"""
TruthLens AI — Evidence Ranker & Stance Detection Module
Scores and ranks retrieved evidence passages relative to extracted claims.
Detects supporting vs. contradicting signals and estimates passage relevance.
"""

import re
import math
from typing import List, Tuple
from backend.schemas.response import EvidenceItem

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further",
    "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his",
    "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me", "more", "most", "my",
    "myself", "no", "nor", "not", "now", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should", "so", "some", "such",
    "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they",
    "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "were",
    "what", "when", "where", "which", "while", "who", "whom", "why", "with", "would", "you", "your"
}

CONTRADICTION_PATTERNS = [
    r"\b(?:false|hoax|untrue|fabricated|debunked|myth|disproved|refuted)\b",
    r"\b(?:no evidence|lack(?:s|ing)? of evidence|unfounded|baseless)\b",
    r"\b(?:denied|rejected|retracted|disputed|misleading|inaccurate)\b",
    r"\b(?:never happened|did not occur|conspiracy theory)\b"
]

SUPPORTING_PATTERNS = [
    r"\b(?:confirmed|verified|proven|demonstrated|corroborated|established)\b",
    r"\b(?:official(?:ly)?|documented|authenticated|recorded|validated)\b",
    r"\b(?:accord(?:ing)? to records|evidence shows|data confirms)\b",
    r"\b(?:announced on|signed into law|passed on)\b"
]


class EvidenceRanker:
    def __init__(self):
        pass

    def rank_evidence(
        self,
        claim_text: str,
        evidence_items: List[EvidenceItem]
    ) -> Tuple[List[EvidenceItem], List[EvidenceItem]]:
        """
        Ranks retrieved evidence and splits into supporting and contradicting lists.
        Returns: (supporting_evidence, contradicting_evidence)
        """
        claim_tokens = self._tokenize(claim_text)
        ranked_items: List[EvidenceItem] = []

        for item in evidence_items:
            passage_text = f"{item.title} {item.snippet}"
            passage_tokens = self._tokenize(passage_text)

            # 1. Lexical overlap & relevance
            relevance = self._calculate_relevance(claim_tokens, passage_tokens, claim_text, passage_text)
            item.relevance_score = round(relevance, 4)

            # 2. Stance detection
            item.stance = self._detect_stance(claim_text, passage_text)
            ranked_items.append(item)

        # Sort all items by relevance descending
        ranked_items.sort(key=lambda x: x.relevance_score, reverse=True)

        supporting = [item for item in ranked_items if item.stance == "SUPPORTING"]
        contradicting = [item for item in ranked_items if item.stance == "CONTRADICTING"]
        neutral = [item for item in ranked_items if item.stance in ["NEUTRAL", "INCONCLUSIVE"]]

        # Attribute relevant neutral items to supporting if no direct refutation
        for n in neutral:
            if n.relevance_score >= 0.35:
                supporting.append(n)

        # If supporting is still empty but we have retrieved items, keep top relevant neutral items
        if not supporting and not contradicting and neutral:
            for n in neutral[:3]:
                if n.relevance_score >= 0.20:
                    supporting.append(n)

        # Ensure top evidence is capped to avoid overwhelming UI
        return supporting[:5], contradicting[:5]

    def _tokenize(self, text: str) -> List[str]:
        words = re.findall(r"\b[a-z0-9]+\b", text.lower())
        return [w for w in words if w not in STOPWORDS and len(w) > 2]

    def _calculate_relevance(
        self,
        claim_tokens: List[str],
        passage_tokens: List[str],
        raw_claim: str,
        raw_passage: str
    ) -> float:
        if not claim_tokens or not passage_tokens:
            return 0.1

        set_claim = set(claim_tokens)
        set_passage = set(passage_tokens)

        overlap = set_claim.intersection(set_passage)
        overlap_score = len(overlap) / len(set_claim)

        # Substring / phrase match bonus
        phrase_bonus = 0.0
        # Check 2-3 word sequences
        claim_words = raw_claim.lower().split()
        if len(claim_words) >= 3:
            for i in range(len(claim_words) - 2):
                trigram = " ".join(claim_words[i:i+3])
                if trigram in raw_passage.lower():
                    phrase_bonus = 0.25
                    break

        raw_score = 0.7 * overlap_score + phrase_bonus + 0.1
        # Normalize between 0.1 and 0.99
        return min(max(raw_score, 0.1), 0.99)

    def _detect_stance(self, claim: str, passage: str) -> str:
        text = passage.lower()

        # Check contradiction first
        for pat in CONTRADICTION_PATTERNS:
            if re.search(pat, text):
                return "CONTRADICTING"

        # Check supporting patterns
        for pat in SUPPORTING_PATTERNS:
            if re.search(pat, text):
                return "SUPPORTING"

        return "NEUTRAL"


_ranker_instance = None


def get_evidence_ranker() -> EvidenceRanker:
    global _ranker_instance
    if _ranker_instance is None:
        _ranker_instance = EvidenceRanker()
    return _ranker_instance
