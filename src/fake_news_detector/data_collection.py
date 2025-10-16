"""Tools for collecting financial and economic news articles from Vietnamese sources."""

from __future__ import annotations

import datetime as _dt
import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import feedparser
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


@dataclass
class Article:
    """Represents a scraped news article."""

    source: str
    title: str
    link: str
    summary: str
    published: Optional[_dt.datetime] = None
    content: Optional[str] = None


NEWS_SOURCES: Dict[str, Dict[str, str]] = {
    "vnexpress": {
        "name": "VnExpress Kinh doanh",
        "rss": "https://vnexpress.net/rss/kinh-doanh.rss",
    },
    "cafef": {
        "name": "CafeF",
        "rss": "https://cafef.vn/tri-thuc/kinh-te.rss",
    },
    "vietstock": {
        "name": "Vietstock",
        "rss": "https://vietstock.vn/rss/",
    },
    "thoibaonganhang": {
        "name": "Thời báo Ngân hàng",
        "rss": "https://thoibaonganhang.vn/rss/tai-chinh",
    },
    "ndh": {
        "name": "NDH",
        "rss": "https://ndh.vn/kinh-te.rss",
    },
}
"""Mapping of short source keys to metadata about each news source."""


def _parse_datetime(entry: feedparser.FeedParserDict) -> Optional[_dt.datetime]:
    """Parse the published datetime from a feed entry if available."""

    published = entry.get("published_parsed") or entry.get("updated_parsed")
    if not published:
        return None
    try:
        return _dt.datetime.fromtimestamp(
            _dt.datetime(*published[:6]).timestamp(), tz=_dt.timezone.utc
        )
    except (TypeError, ValueError):
        logger.debug("Could not parse published date for entry %s", entry.get("link"))
        return None


def fetch_articles(source_key: str, limit: int = 20) -> List[Article]:
    """Fetch the latest articles from a single RSS source.

    Parameters
    ----------
    source_key:
        Key of the source as defined in :data:`NEWS_SOURCES`.
    limit:
        Maximum number of items to return from the feed.
    """

    source = NEWS_SOURCES.get(source_key)
    if not source:
        raise KeyError(f"Unknown news source: {source_key}")

    feed = feedparser.parse(source["rss"])
    articles: List[Article] = []

    for entry in feed.entries[:limit]:
        article = Article(
            source=source_key,
            title=entry.get("title", ""),
            link=entry.get("link", ""),
            summary=entry.get("summary", ""),
            published=_parse_datetime(entry),
        )
        articles.append(article)

    return articles


def enrich_article_content(article: Article, timeout: int = 10) -> Article:
    """Download and populate the full HTML content for an article."""

    if not article.link:
        return article

    try:
        response = requests.get(article.link, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:  # pragma: no cover - network failures
        logger.warning("Failed to download article %s: %s", article.link, exc)
        return article

    soup = BeautifulSoup(response.text, "html.parser")
    # Heuristic: get main article content by selecting long text paragraphs
    paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
    article.content = "\n".join(p for p in paragraphs if len(p) > 40)
    if not article.content:
        article.content = soup.get_text("\n")
    return article


def collect_latest_articles(
    sources: Optional[Iterable[str]] = None,
    limit_per_source: int = 20,
    enrich: bool = False,
) -> List[Article]:
    """Collect the latest articles from one or more sources.

    Parameters
    ----------
    sources:
        Iterable of source keys to fetch. Defaults to all sources.
    limit_per_source:
        Number of articles to fetch per source.
    enrich:
        Whether to download the article body.
    """

    if sources is None:
        sources = NEWS_SOURCES.keys()

    seen_links = set()
    all_articles: List[Article] = []

    for key in sources:
        try:
            articles = fetch_articles(key, limit=limit_per_source)
        except Exception as exc:  # pragma: no cover - network variation
            logger.error("Failed to fetch articles for %s: %s", key, exc)
            continue

        for article in articles:
            if article.link in seen_links:
                continue
            seen_links.add(article.link)

            if enrich:
                article = enrich_article_content(article)
            all_articles.append(article)

    # Sort by published date descending if available
    all_articles.sort(
        key=lambda a: a.published or _dt.datetime.min.replace(tzinfo=_dt.timezone.utc),
        reverse=True,
    )
    return all_articles


__all__ = [
    "Article",
    "collect_latest_articles",
    "enrich_article_content",
    "fetch_articles",
    "NEWS_SOURCES",
]
