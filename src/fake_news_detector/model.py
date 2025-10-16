"""Machine learning model wrapper for fake news detection."""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .preprocessing import clean_text


@dataclass
class TrainingConfig:
    """Configuration for training the fake news detector."""

    test_size: float = 0.2
    random_state: int = 42
    max_features: int = 5000
    ngram_range: tuple[int, int] = (1, 2)
    penalty: str = "l2"
    C: float = 1.0


@dataclass
class FakeNewsDetector:
    """Wrapper around a scikit-learn pipeline for fake news detection."""

    config: TrainingConfig = field(default_factory=TrainingConfig)
    pipeline: Optional[Pipeline] = None

    def build_pipeline(self) -> Pipeline:
        """Construct the scikit-learn pipeline."""

        vectorizer = TfidfVectorizer(
            max_features=self.config.max_features,
            ngram_range=self.config.ngram_range,
            preprocessor=clean_text,
        )
        classifier = LogisticRegression(
            penalty=self.config.penalty,
            C=self.config.C,
            max_iter=1000,
            n_jobs=None,
        )
        self.pipeline = Pipeline(
            [
                ("vectorizer", vectorizer),
                ("classifier", classifier),
            ]
        )
        return self.pipeline

    def train(
        self,
        texts: Iterable[str],
        labels: Iterable[int],
        validation_split: Optional[float] = None,
    ) -> dict:
        """Train the model and optionally report validation metrics."""

        if self.pipeline is None:
            self.build_pipeline()

        texts = list(texts)
        labels = list(labels)

        metrics: dict = {}
        if validation_split is None:
            validation_split = self.config.test_size

        if validation_split > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                texts,
                labels,
                test_size=validation_split,
                random_state=self.config.random_state,
                stratify=labels,
            )
            self.pipeline.fit(X_train, y_train)
            predictions = self.pipeline.predict(X_val)
            metrics = classification_report(
                y_val,
                predictions,
                output_dict=True,
                zero_division=0,
            )
        else:
            self.pipeline.fit(texts, labels)
        return metrics

    def predict(self, texts: Iterable[str]) -> List[int]:
        """Predict labels for the given texts."""

        if self.pipeline is None:
            raise RuntimeError("Model pipeline has not been built or loaded.")
        return list(self.pipeline.predict(list(texts)))

    def save(self, path: Path) -> None:
        """Persist the trained pipeline to disk."""

        if self.pipeline is None:
            raise RuntimeError("Nothing to save. Train the model first.")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"config": self.config, "pipeline": self.pipeline}, path)

    @classmethod
    def load(cls, path: Path) -> "FakeNewsDetector":
        """Load a pipeline from disk."""

        data = joblib.load(path)
        detector = cls(config=data["config"])
        detector.pipeline = data["pipeline"]
        return detector

    @staticmethod
    def from_dataframe(
        frame: pd.DataFrame,
        text_column: str = "content",
        label_column: str = "label",
    ) -> tuple[List[str], List[int]]:
        """Extract texts and labels from a pandas dataframe."""

        texts = frame[text_column].fillna("").astype(str).tolist()
        labels = frame[label_column].astype(int).tolist()
        return texts, labels


__all__ = ["FakeNewsDetector", "TrainingConfig"]
