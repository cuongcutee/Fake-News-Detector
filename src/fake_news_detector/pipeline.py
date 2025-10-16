"""End-to-end workflow helpers for collecting data and training models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
from tqdm import tqdm

from .data_collection import Article, collect_latest_articles
from .model import FakeNewsDetector


def articles_to_dataframe(articles: Iterable[Article]) -> pd.DataFrame:
    """Convert a list of articles to a pandas dataframe."""

    records = []
    for article in articles:
        records.append(
            {
                "source": article.source,
                "title": article.title,
                "summary": article.summary,
                "link": article.link,
                "published": article.published.isoformat()
                if article.published
                else None,
                "content": article.content or article.summary,
            }
        )
    return pd.DataFrame(records)


def collect_and_save(
    output_path: Path,
    sources: Optional[Iterable[str]] = None,
    limit_per_source: int = 20,
    enrich: bool = False,
) -> pd.DataFrame:
    """Collect latest articles and save them to disk as JSON."""

    articles = collect_latest_articles(
        sources, limit_per_source=limit_per_source, enrich=enrich
    )
    frame = articles_to_dataframe(articles)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_json(output_path, orient="records", force_ascii=False, indent=2)
    return frame


def train_from_dataset(
    dataset_path: Path,
    text_column: str,
    label_column: str,
    model_output: Path,
    config_path: Optional[Path] = None,
) -> dict:
    """Train the fake news detector using a dataset stored on disk."""

    frame = pd.read_csv(dataset_path)
    detector = FakeNewsDetector()
    texts, labels = FakeNewsDetector.from_dataframe(
        frame,
        text_column=text_column,
        label_column=label_column,
    )
    metrics = detector.train(texts, labels)
    detector.save(model_output)

    if config_path:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(detector.config.__dict__, indent=2))
    return metrics


def batch_predict(
    model_path: Path,
    input_path: Path,
    text_column: str = "content",
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Generate predictions for a dataset using a trained model."""

    detector = FakeNewsDetector.load(model_path)
    frame = pd.read_csv(input_path)
    texts = frame[text_column].fillna("").astype(str)
    predictions = detector.predict(texts)
    frame["prediction"] = predictions

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_path, index=False)

    return frame


def interactive_labeling(
    articles: Iterable[Article],
    label_output: Path,
    true_label_prompt: str = "Nhập 1 nếu tin thật,",
    false_label_prompt: str = "Nhập 0 nếu tin giả,",
) -> None:
    """Simple terminal-based labeling helper for building datasets."""

    records = []
    for article in tqdm(list(articles), desc="Labeling"):
        print("\n" + article.title)
        print(article.summary)
        while True:
            label = input(
                f"{true_label_prompt} {false_label_prompt} hoặc s để bỏ qua: "
            ).strip()
            if label in {"0", "1"}:
                records.append(
                    {
                        "title": article.title,
                        "summary": article.summary,
                        "content": article.content or article.summary,
                        "label": int(label),
                        "source": article.source,
                    }
                )
                break
            if label.lower() == "s":
                break
            print("Lựa chọn không hợp lệ, vui lòng thử lại.")

    if records:
        label_output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_csv(label_output, index=False)


__all__ = [
    "articles_to_dataframe",
    "batch_predict",
    "collect_and_save",
    "interactive_labeling",
    "train_from_dataset",
]
