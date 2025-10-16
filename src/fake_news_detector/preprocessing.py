"""Text preprocessing helpers tailored for Vietnamese financial news."""

from __future__ import annotations

import re
from typing import Iterable, List

import pandas as pd

_extra_space_re = re.compile(r"\s{2,}")

_stopwords = {
    "là",
    "của",
    "và",
    "có",
    "cho",
    "với",
    "một",
    "các",
    "những",
    "được",
    "từ",
    "khi",
    "đã",
    "sẽ",
}


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace into a single space."""

    text = _extra_space_re.sub(" ", text)
    return text.strip()


def strip_punctuation(text: str) -> str:
    """Remove punctuation except for periods to retain sentence structure."""

    text = re.sub(r"[^0-9A-Za-zÀ-ỹ\. ]+", " ", text)
    return normalize_whitespace(text)


def remove_stopwords(tokens: Iterable[str]) -> List[str]:
    """Remove a predefined list of Vietnamese stopwords."""

    return [token for token in tokens if token.lower() not in _stopwords]


def clean_text(text: str) -> str:
    """Normalize, remove punctuation and stopwords from text."""

    text = text.lower()
    text = strip_punctuation(text)
    tokens = text.split()
    tokens = remove_stopwords(tokens)
    return " ".join(tokens)


def preprocess_dataframe(
    frame: pd.DataFrame,
    text_column: str = "content",
    cleaned_column: str = "cleaned_content",
) -> pd.DataFrame:
    """Return a copy of the dataframe with an additional cleaned text column."""

    frame = frame.copy()
    frame[cleaned_column] = frame[text_column].fillna("").map(clean_text)
    return frame


__all__ = [
    "clean_text",
    "normalize_whitespace",
    "preprocess_dataframe",
    "remove_stopwords",
    "strip_punctuation",
]
