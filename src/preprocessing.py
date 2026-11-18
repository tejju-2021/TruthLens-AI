"""
TruthLens AI — Data Preprocessing Utilities
Re-exports preprocessing routines for pipeline training and evaluation.
"""

from backend.utils.preprocessing import clean_text, remove_source_cues, truncate_text

__all__ = ["clean_text", "remove_source_cues", "truncate_text"]
