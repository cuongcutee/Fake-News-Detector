"""Top-level package for the Vietnamese financial news fake news detector."""

from .data_collection import Article, NEWS_SOURCES, collect_latest_articles
from .preprocessing import clean_text
from .model import FakeNewsDetector

__all__ = [
    "Article",
    "collect_latest_articles",
    "NEWS_SOURCES",
    "clean_text",
    "FakeNewsDetector",
]
