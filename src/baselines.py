"""Non-neural baseline: TF-IDF features + linear SVM for binary toxic detection.

Provides the same ``predict_toxic_proba`` interface as the neural scorers in
``model_registry.py`` so the comparison treats it identically. This is the
"non-neural floor" the proposal calls for — a fair-comparison anchor showing
how much (if anything) the transformers buy over a strong classical model.

LinearSVC has no native probabilities, so we wrap it in CalibratedClassifierCV
(Platt scaling) to obtain calibrated P(toxic) — also useful for the later
calibration analysis.
"""
from typing import Optional, Sequence

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from paths import SEED


class TfidfSvmScorer:
    """TF-IDF (word 1–2 grams) + calibrated LinearSVM, binary toxic vs not."""

    model_id = "tfidf+linearsvm"

    def __init__(
        self,
        max_features: int = 50_000,
        ngram_range: tuple = (1, 2),
        C: float = 1.0,
        seed: int = SEED,
    ):
        self.pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=max_features,
                        ngram_range=ngram_range,
                        sublinear_tf=True,
                        strip_accents="unicode",
                        min_df=2,
                    ),
                ),
                (
                    "svm",
                    CalibratedClassifierCV(
                        # dual=True suits text (n_features > n_samples); set
                        # explicitly to silence the sklearn 1.5 default change.
                        estimator=LinearSVC(C=C, dual=True, random_state=seed),
                        cv=3,
                    ),
                ),
            ]
        )
        # Rough capacity proxy for fair-comparison tables (vocabulary size).
        self.n_params: Optional[int] = None

    def fit(self, texts: Sequence[str], y: Sequence[int]) -> "TfidfSvmScorer":
        self.pipeline.fit(list(texts), np.asarray(y).astype(int))
        self.n_params = int(len(self.pipeline.named_steps["tfidf"].vocabulary_))
        return self

    def predict_toxic_proba(self, texts: Sequence[str], batch_size: Optional[int] = None) -> np.ndarray:
        """Return calibrated P(toxic). ``batch_size`` is ignored (kept for interface parity)."""
        if self.n_params is None:
            raise RuntimeError("TfidfSvmScorer must be .fit() before scoring.")
        return self.pipeline.predict_proba(list(texts))[:, 1]
