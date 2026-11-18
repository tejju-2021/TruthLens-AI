"""
TruthLens AI — Evidence Retrieval Module
Modular evidence retrieval system supporting multiple real-world providers:
- Wikipedia API (Public encyclopedic facts)
- DuckDuckGo / Web search (Recent news, institutional reports)
- MultiSource aggregator with intelligent query refinement

Preserves real source URLs, titles, snippets, and domains.
Never fabricates sources or links.
"""

import re
import urllib.parse
import logging
from typing import List, Optional
import httpx

from backend.schemas.response import EvidenceItem
from backend.config import settings

logger = logging.getLogger("truthlens.evidence_retriever")

USER_AGENT = "TruthLensAI/2.0 (Fact Verification Research Pipeline; truthlens@example.org)"

COMMON_FILLERS = {
    "a", "an", "the", "is", "are", "was", "were", "and", "or", "by", "for", "with",
    "about", "against", "between", "into", "through", "during", "before", "after",
    "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over",
    "under", "again", "further", "then", "once", "here", "there", "when", "where",
    "why", "how", "all", "any", "both", "each", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
    "very", "can", "will", "just", "don", "should", "now", "completely", "surrounded"
}


def clean_html_tags(raw_html: str) -> str:
    """Removes HTML markup from snippets."""
    clean = re.sub(r"<[^>]+>", " ", raw_html)
    clean = re.sub(r"\s+", " ", clean)
    return clean.strip()


def extract_domain(url: Optional[str]) -> str:
    """Extracts top-level domain from URL."""
    if not url:
        return "unknown"
    try:
        parsed = urllib.parse.urlparse(url)
        netloc = parsed.netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc
    except Exception:
        return "unknown"


def extract_salient_keywords(query: str, max_words: int = 7) -> str:
    """Extracts salient content keywords for search queries."""
    words = re.findall(r"\b[A-Za-z0-9'-]+\b", query)
    salient = [w for w in words if w.lower() not in COMMON_FILLERS and len(w) > 2]
    if not salient:
        return query
    return " ".join(salient[:max_words])


class WikipediaRetriever:
    """Retrieves factual summaries and passages from Wikipedia's public API."""
    API_URL = "https://en.wikipedia.org/w/api.php"

    def search(self, query: str, max_results: int = 4) -> List[EvidenceItem]:
        # Try full query first
        hits = self._execute_query(query, max_results)
        # If fewer than 2 hits and query is long, try keyword query
        if len(hits) < 2 and len(query.split()) > 5:
            keyword_query = extract_salient_keywords(query)
            if keyword_query != query:
                extra_hits = self._execute_query(keyword_query, max_results)
                seen_titles = {h.title for h in hits}
                for h in extra_hits:
                    if h.title not in seen_titles:
                        hits.append(h)
                        if len(hits) >= max_results:
                            break
        return hits

    def _execute_query(self, query: str, max_results: int) -> List[EvidenceItem]:
        evidence_list: List[EvidenceItem] = []
        try:
            params = {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "utf8": 1,
                "srlimit": max_results,
            }
            headers = {"User-Agent": USER_AGENT}
            with httpx.Client(timeout=settings.RETRIEVAL_TIMEOUT_SECONDS) as client:
                resp = client.get(self.API_URL, params=params, headers=headers)
                if resp.status_code != 200:
                    logger.warning(f"Wikipedia search returned status {resp.status_code}")
                    return []
                data = resp.json()

            search_hits = data.get("query", {}).get("search", [])
            for i, hit in enumerate(search_hits):
                title = hit.get("title", "")
                snippet = clean_html_tags(hit.get("snippet", ""))
                encoded_title = urllib.parse.quote(title.replace(" ", "_"))
                url = f"https://en.wikipedia.org/wiki/{encoded_title}"

                evidence_list.append(
                    EvidenceItem(
                        id=i + 1,
                        title=f"Wikipedia: {title}",
                        snippet=snippet,
                        source_url=url,
                        domain="wikipedia.org",
                        stance="NEUTRAL",
                        relevance_score=0.8,
                        published_date=hit.get("timestamp", None),
                        credibility_tier="Reputable Reference",
                    )
                )
        except Exception as e:
            logger.warning(f"Wikipedia retrieval error for query '{query}': {e}")
        return evidence_list


class DuckDuckGoRetriever:
    """Retrieves web search snippets from DuckDuckGo Instant Answer / HTML API."""
    IA_API_URL = "https://api.duckduckgo.com/"
    HTML_SEARCH_URL = "https://html.duckduckgo.com/html/"

    def search(self, query: str, max_results: int = 4) -> List[EvidenceItem]:
        evidence_list: List[EvidenceItem] = []
        try:
            # 1. First attempt Instant Answer API
            params = {
                "q": query,
                "format": "json",
                "no_html": "1",
                "skip_disambig": "1",
            }
            headers = {"User-Agent": USER_AGENT}
            with httpx.Client(timeout=settings.RETRIEVAL_TIMEOUT_SECONDS, follow_redirects=True) as client:
                resp = client.get(self.IA_API_URL, params=params, headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    abstract = data.get("AbstractText", "")
                    abstract_url = data.get("AbstractURL", "")
                    abstract_source = data.get("AbstractSource", "Web Reference")
                    heading = data.get("Heading", query)

                    if abstract:
                        evidence_list.append(
                            EvidenceItem(
                                id=1,
                                title=f"{abstract_source}: {heading}",
                                snippet=abstract,
                                source_url=abstract_url or "https://duckduckgo.com/?q=" + urllib.parse.quote(query),
                                domain=extract_domain(abstract_url),
                                stance="NEUTRAL",
                                relevance_score=0.85,
                                credibility_tier="Reputable Reference",
                            )
                        )

            # 2. If IA didn't yield enough, query HTML search
            if len(evidence_list) < max_results:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                }
                search_query = query if len(query.split()) <= 8 else extract_salient_keywords(query)
                with httpx.Client(timeout=settings.RETRIEVAL_TIMEOUT_SECONDS, follow_redirects=True) as client:
                    resp = client.post(
                        self.HTML_SEARCH_URL,
                        data={"q": search_query, "b": ""},
                        headers=headers
                    )
                    if resp.status_code == 200:
                        html = resp.text
                        result_blocks = re.findall(
                            r'<a class="result__url" href="([^"]+)".*?<a class="result__snippet[^>]*>(.*?)</a>',
                            html,
                            re.DOTALL
                        )
                        for raw_url, raw_snippet in result_blocks:
                            if len(evidence_list) >= max_results:
                                break
                            clean_snippet = clean_html_tags(raw_snippet)
                            url = raw_url
                            if "uddg=" in raw_url:
                                match = re.search(r"uddg=([^&]+)", raw_url)
                                if match:
                                    url = urllib.parse.unquote(match.group(1))

                            domain = extract_domain(url)
                            if not clean_snippet or domain in ["duckduckgo.com", "unknown"]:
                                continue

                            evidence_list.append(
                                EvidenceItem(
                                    id=len(evidence_list) + 1,
                                    title=f"Search Result ({domain})",
                                    snippet=clean_snippet,
                                    source_url=url,
                                    domain=domain,
                                    stance="NEUTRAL",
                                    relevance_score=0.75,
                                    credibility_tier="General Web",
                                )
                            )
        except Exception as e:
            logger.warning(f"DuckDuckGo retrieval error for query '{query}': {e}")
        return evidence_list


class MultiSourceRetriever:
    """Aggregates evidence across Wikipedia and Web Search."""
    def __init__(self):
        self.wiki = WikipediaRetriever()
        self.ddg = DuckDuckGoRetriever()

    def search(self, query: str, max_results: int = 5) -> List[EvidenceItem]:
        evidence: List[EvidenceItem] = []
        seen_urls = set()

        # 1. Query Wikipedia (high factual credibility)
        wiki_hits = self.wiki.search(query, max_results=3)
        for item in wiki_hits:
            if item.source_url not in seen_urls:
                seen_urls.add(item.source_url)
                evidence.append(item)

        # 2. Query DuckDuckGo for live external coverage
        remaining = max_results - len(evidence)
        if remaining > 0:
            ddg_hits = self.ddg.search(query, max_results=remaining + 2)
            for item in ddg_hits:
                if item.source_url not in seen_urls:
                    seen_urls.add(item.source_url)
                    evidence.append(item)
                if len(evidence) >= max_results:
                    break

        for i, item in enumerate(evidence):
            item.id = i + 1

        return evidence


_retriever_instance = None


def get_evidence_retriever():
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = MultiSourceRetriever()
    return _retriever_instance
