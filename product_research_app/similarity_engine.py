"""Local similarity search to reuse enrichment scores for near-duplicate items."""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize

from .student_model import build_feature_sample, MappingLike


logger = logging.getLogger(__name__)


def _clamp(value: Optional[float]) -> Optional[int]:
    if value is None:
        return None
    return max(0, min(100, int(round(value))))


@dataclass
class SimilarityMatch:
    desire: Optional[int]
    awareness: Optional[int]
    reason: str
    score: float
    source: str
    reference_sig: Optional[str] = None


class SimilarityEngine:
    """Lightweight in-memory similarity matcher backed by TF-IDF style hashing."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        threshold: float = 0.88,
        max_entries: int = 5000,
        logger: Optional[logging.Logger] = None,
        n_features: int = 2 ** 15,
    ) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.threshold = float(threshold)
        self.max_entries = int(max_entries)
        self.enabled = enabled
        self.vectorizer = HashingVectorizer(
            n_features=n_features,
            alternate_sign=False,
            ngram_range=(1, 2),
        )
        self._records: List[Dict[str, Any]] = []
        self._embeddings: Optional[sparse.csr_matrix] = None
        self._lock = threading.Lock()

    def prepare(self, rows: Sequence[Any]) -> None:
        if not self.enabled:
            return
        texts: List[str] = []
        metadata: List[Dict[str, Any]] = []
        for row in rows:
            try:
                raw = json.loads(row["raw"])
            except Exception:
                raw = {}
            if not raw:
                continue
            try:
                result = json.loads(row["result"])
            except Exception:
                result = {}
            if not isinstance(result, dict):
                continue
            sample = build_feature_sample(raw)
            text = str(sample.get("text") or "").strip()
            if not text:
                continue
            metadata.append(
                {
                    "sig_hash": row["sig_hash"],
                    "desire": result.get("desire"),
                    "awareness": result.get("awareness"),
                    "reason": result.get("reason"),
                    "source": result.get("source") or "ai",
                }
            )
            texts.append(text)
            if len(texts) >= self.max_entries:
                break
        if not texts:
            return
        matrix = self.vectorizer.transform(texts)
        embeddings = normalize(matrix, norm="l2", axis=1)
        with self._lock:
            self._records = metadata
            self._embeddings = embeddings.tocsr()
        self.logger.info("Similarity index primed with %d records", len(self._records))

    def match(self, sample: MappingLike, *, sig_hash: Optional[str] = None) -> Optional[SimilarityMatch]:
        if not self.enabled or self._embeddings is None or not self._records:
            return None
        text = str(sample.get("text") or "").strip()
        if not text:
            return None
        vector = self.vectorizer.transform([text])
        vector = normalize(vector, norm="l2", axis=1)
        with self._lock:
            embeddings = self._embeddings
            records = list(self._records)
        if embeddings is None or not records:
            return None
        scores = embeddings @ vector.T
        if scores.shape[0] == 0:
            return None
        scores_dense = scores.toarray().ravel()
        if scores_dense.size == 0:
            return None
        idx = int(np.argmax(scores_dense))
        score = float(scores_dense[idx])
        if score < self.threshold:
            return None
        record = records[idx]
        if record.get("sig_hash") == sig_hash:
            return None
        base_reason = record.get("reason") or "Hereda de similar"
        reason = f"{base_reason} · sim {score:.2f}"[:120]
        return SimilarityMatch(
            desire=_clamp(record.get("desire")),
            awareness=_clamp(record.get("awareness")),
            reason=reason,
            score=score,
            source="similarity",
            reference_sig=record.get("sig_hash"),
        )

    def register(self, sample: MappingLike, result: MappingLike, *, sig_hash: Optional[str] = None) -> None:
        if not self.enabled:
            return
        text = str(sample.get("text") or "").strip()
        if not text:
            return
        vector = self.vectorizer.transform([text])
        vector = normalize(vector, norm="l2", axis=1)
        record = {
            "sig_hash": sig_hash,
            "desire": result.get("desire"),
            "awareness": result.get("awareness"),
            "reason": result.get("reason"),
            "source": result.get("source") or "student",
        }
        with self._lock:
            if self._embeddings is None:
                self._embeddings = vector.tocsr()
                self._records = [record]
            else:
                if sig_hash:
                    for idx, existing in enumerate(self._records):
                        if existing.get("sig_hash") == sig_hash:
                            self._records.pop(idx)
                            self._embeddings = sparse.vstack(
                                [self._embeddings[:idx], self._embeddings[idx + 1 :]]
                            )
                            break
                self._embeddings = sparse.vstack([self._embeddings, vector])
                self._records.append(record)
                if len(self._records) > self.max_entries:
                    self._records = self._records[-self.max_entries :]
                    self._embeddings = self._embeddings[-self.max_entries :, :]
        self.logger.debug(
            "Similarity index updated (size=%d, sig=%s)",
            len(self._records),
            sig_hash,
        )

