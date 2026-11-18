import re
from typing import Optional

NOISE_PREFIXES = r"^(?:advertisement|sponsored|breaking(?:\s+news)?|live(?:\s+updates)?|updated|watch|exclusive|opinion)\s*[:\-–—]?\s*"


def clean_text(text: Optional[str]) -> str:
    """Normalize whitespace and strip text, removing common web copy-paste prefixes."""
    if not text:
        return ""
    text = str(text)
    # Strip web copy-paste boilerplate prefixes (e.g. "Advertisement", "Breaking News:")
    text = re.sub(NOISE_PREFIXES, "", text.strip(), flags=re.IGNORECASE)
    # Normalize multiple whitespace characters
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def remove_source_cues(text: str) -> str:
    """
    Remove common news wire prefixes like 'WASHINGTON (Reuters) - '
    observed in ISOT dataset to test robustness against stylistic leakage.
    """
    text = str(text)
    # Match patterns like:
    # "WASHINGTON (Reuters) - "
    # "LONDON (Reuters) - "
    # "NEW YORK (AP) - "
    pattern = r"^[A-Z\s]+(?:\s*\([A-Za-z\s]+\))?\s*[-–—]\s*"
    cleaned = re.sub(pattern, "", text.strip())
    return cleaned.strip()


def truncate_text(text: str, max_chars: int = 2000) -> str:
    """Truncate text cleanly at word boundary if it exceeds max_chars."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")
    if last_space > 0:
        truncated = truncated[:last_space]
    return truncated + "..."
