"""Utilities for training and serving the local "student" enrichment model.

The student model supplements the remote AI service by learning from previous
AI-labelled examples.  It consumes lightweight textual and numerical features
and predicts ``desire`` and ``awareness`` scores.  When the confidence is high
enough the worker can skip the expensive AI call altogether.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np

MappingLike = Dict[str, Any]

try:  # pragma: no cover - optional dependency, validated via unit tests
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import FeatureUnion, Pipeline
    from sklearn.preprocessing import FunctionTransformer
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception:  # pragma: no cover - handled gracefully by callers
    LogisticRegression = None  # type: ignore[assignment]
    FeatureUnion = None  # type: ignore[assignment]
    Pipeline = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent / "models"
DESIRE_MODEL_FILE = "student_desire.joblib"
AWARENESS_MODEL_FILE = "student_awareness.joblib"
MIN_TRAIN_SAMPLES = 30

TEXT_FIELDS: tuple[str, ...] = (
    "title",
    "name",
    "product_title",
    "short_title",
    "subtitle",
    "description",
    "short_description",
    "summary",
    "bullet_points",
    "category",
    "subcategory",
    "brand",
    "keywords",
)

DATE_FIELDS: tuple[str, ...] = (
    "launch_date",
    "release_date",
    "first_available",
    "first_seen",
    "created_at",
    "date",
)

@dataclass
class StudentSample:
    """Container for a labelled enrichment example."""

    sig_hash: str
    features: Dict[str, Any]
    desire: int
    awareness: int
    source: str
    item_id: Optional[int] = None
    updated_at: Optional[str] = None


def _to_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", ".")
    if not text:
        return None
    try:
        cleaned = text.replace("€", "").replace("$", "").replace("%", "")
        if cleaned.lower().endswith("k"):
            return float(cleaned[:-1]) * 1_000.0
        if cleaned.lower().endswith("m"):
            return float(cleaned[:-1]) * 1_000_000.0
        return float(cleaned)
    except Exception:
        return None


def _parse_date(value: Any) -> Optional[datetime]:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d", "%Y%m%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _compute_oldness_days(raw: MappingLike) -> Optional[float]:
    for key in DATE_FIELDS:
        value = raw.get(key)
        dt = _parse_date(value)
        if dt is None:
            continue
        try:
            delta = datetime.utcnow() - dt
            return float(delta.days)
        except Exception:
            continue
    return None


def _collect_text(raw: MappingLike, product: Optional[MappingLike]) -> str:
    seen: set[str] = set()
    parts: List[str] = []
    for source in (raw, product or {}):
        for key in TEXT_FIELDS:
            value = source.get(key)
            if not value:
                continue
            text = str(value).strip()
            if not text:
                continue
            lowered = text.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            parts.append(text)
    if not parts and raw.get("name"):
        parts.append(str(raw.get("name")))
    return " ".join(parts)


def build_feature_sample(
    raw: MappingLike,
    product: Optional[MappingLike] = None,
) -> Dict[str, Any]:
    """Return the feature dictionary consumed by the student model."""

    text = _collect_text(raw, product)
    price = _to_float(raw.get("price") or (product or {}).get("price"))
    rating = _to_float(raw.get("rating") or (product or {}).get("rating"))
    units = _to_float(raw.get("units_sold") or raw.get("sold") or (product or {}).get("units_sold"))
    oldness = _compute_oldness_days(raw)
    sample = {
        "text": text,
        "price": price if price is not None else 0.0,
        "rating": rating if rating is not None else 0.0,
        "units_sold": units if units is not None else 0.0,
        "oldness": oldness if oldness is not None else 0.0,
    }
    return sample


def _load_json(text: Optional[str]) -> Dict[str, Any]:
    if not text:
        return {}
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        logger.debug("Failed to parse JSON payload", exc_info=True)
    return {}


def load_training_samples(
    conn: Any,
    *,
    limit: Optional[int] = None,
    sources: Sequence[str] = ("ai",),
) -> List[StudentSample]:
    """Fetch labelled items suitable for student training."""

    allowed = {str(src).lower() for src in sources}
    cur = conn.cursor()
    query = (
        "SELECT id, sig_hash, raw, result, updated_at FROM items "
        "WHERE state='enriched' AND result IS NOT NULL ORDER BY updated_at DESC"
    )
    if limit:
        query += f" LIMIT {int(limit)}"
    cur.execute(query)
    rows = cur.fetchall()
    samples: List[StudentSample] = []
    for row in rows:
        raw_data = _load_json(row["raw"])
        result = _load_json(row["result"])
        if not raw_data or not result:
            continue
        source = str(result.get("source") or "").lower() or "ai"
        if allowed and source not in allowed:
            continue
        try:
            desire = int(result.get("desire"))
            awareness = int(result.get("awareness"))
        except Exception:
            continue
        sample = build_feature_sample(raw_data)
        if not sample.get("text"):
            continue
        samples.append(
            StudentSample(
                sig_hash=row["sig_hash"],
                features=sample,
                desire=desire,
                awareness=awareness,
                source=source,
                item_id=row["id"],
                updated_at=row["updated_at"],
            )
        )
    return samples


def _feature_union(random_state: int) -> Pipeline:
    if FeatureUnion is None or Pipeline is None:
        raise RuntimeError("scikit-learn is required to train the student model")

    def _texts(X: Iterable[MappingLike]) -> List[str]:
        return [str(sample.get("text") or "") for sample in X]

    def _numeric(X: Iterable[MappingLike]) -> List[Dict[str, float]]:
        payload: List[Dict[str, float]] = []
        for sample in X:
            payload.append(
                {
                    "price": float(sample.get("price") or 0.0),
                    "rating": float(sample.get("rating") or 0.0),
                    "units_sold": float(sample.get("units_sold") or 0.0),
                    "oldness": float(sample.get("oldness") or 0.0),
                }
            )
        return payload

    text_pipeline = Pipeline(
        [
            ("extract", FunctionTransformer(_texts, validate=False)),
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=6000,
                    ngram_range=(1, 2),
                    min_df=2,
                ),
            ),
        ]
    )
    numeric_pipeline = Pipeline(
        [
            ("extract", FunctionTransformer(_numeric, validate=False)),
            ("vectorizer", DictVectorizer()),
        ]
    )
    union = FeatureUnion(
        [
            ("text", text_pipeline),
            ("numeric", numeric_pipeline),
        ]
    )
    model = LogisticRegression(
        max_iter=300,
        solver="lbfgs",
        multi_class="multinomial",
        random_state=random_state,
    )
    return Pipeline([("features", union), ("model", model)])


def _evaluate_model(
    pipeline: Pipeline,
    X_test: Sequence[MappingLike],
    y_test: Sequence[int],
    *,
    threshold: float,
) -> Dict[str, Any]:
    predictions = pipeline.predict(X_test)
    try:
        proba = pipeline.predict_proba(X_test)
        confidences = np.max(proba, axis=1)
    except Exception:
        confidences = np.zeros(len(predictions))
    mae = mean_absolute_error(y_test, predictions) if len(y_test) else 0.0
    r2 = r2_score(y_test, predictions) if len(y_test) > 1 else 0.0
    coverage = float(np.mean(confidences >= threshold)) if len(confidences) else 0.0
    return {
        "mae": float(mae),
        "r2": float(r2),
        "coverage": coverage,
    }


def train_student_models(
    conn: Any,
    *,
    limit: Optional[int] = None,
    confidence_threshold: float = 0.65,
    model_dir: Optional[Path] = None,
    sources: Sequence[str] = ("ai",),
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train both desire and awareness student models and persist them."""

    if LogisticRegression is None:
        raise RuntimeError("scikit-learn is required to train the student model")

    samples = load_training_samples(conn, limit=limit, sources=sources)
    if len(samples) < MIN_TRAIN_SAMPLES:
        raise ValueError(
            f"Insufficient samples for training: {len(samples)} available, "
            f"need at least {MIN_TRAIN_SAMPLES}"
        )

    X = [sample.features for sample in samples]
    y_desire = np.array([sample.desire for sample in samples], dtype=int)
    y_awareness = np.array([sample.awareness for sample in samples], dtype=int)

    splits = train_test_split(
        X,
        y_desire,
        y_awareness,
        test_size=0.2,
        random_state=random_state,
    )
    X_train, X_test, y_desire_train, y_desire_test, y_awareness_train, y_awareness_test = splits

    if len(np.unique(y_desire_train)) < 2 or len(np.unique(y_awareness_train)) < 2:
        raise ValueError("Need at least two classes to train the student model")

    desire_pipeline = _feature_union(random_state)
    awareness_pipeline = _feature_union(random_state + 7)

    desire_pipeline.fit(X_train, y_desire_train)
    awareness_pipeline.fit(X_train, y_awareness_train)

    desire_metrics = _evaluate_model(
        desire_pipeline, X_test, y_desire_test, threshold=confidence_threshold
    )
    awareness_metrics = _evaluate_model(
        awareness_pipeline, X_test, y_awareness_test, threshold=confidence_threshold
    )

    directory = model_dir or MODEL_DIR
    directory.mkdir(parents=True, exist_ok=True)
    trained_at = datetime.utcnow().isoformat()

    joblib.dump(
        {
            "pipeline": desire_pipeline,
            "trained_at": trained_at,
            "metrics": desire_metrics,
            "confidence_threshold": confidence_threshold,
            "target": "desire",
            "n_samples": len(samples),
        },
        directory / DESIRE_MODEL_FILE,
    )
    joblib.dump(
        {
            "pipeline": awareness_pipeline,
            "trained_at": trained_at,
            "metrics": awareness_metrics,
            "confidence_threshold": confidence_threshold,
            "target": "awareness",
            "n_samples": len(samples),
        },
        directory / AWARENESS_MODEL_FILE,
    )

    return {
        "trained_at": trained_at,
        "samples": len(samples),
        "confidence_threshold": confidence_threshold,
        "desire": desire_metrics,
        "awareness": awareness_metrics,
    }


class StudentModelManager:
    """Load and serve predictions from the student model on demand."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        confidence_threshold: float = 0.65,
        model_dir: Optional[Path | str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.enabled = enabled
        self.confidence_threshold = confidence_threshold
        self.model_dir = Path(model_dir) if model_dir else MODEL_DIR
        self.desire_model: Optional[Dict[str, Any]] = None
        self.awareness_model: Optional[Dict[str, Any]] = None
        self._loaded = False
        if self.enabled:
            self._load_models()

    @property
    def is_ready(self) -> bool:
        return self.enabled and self._loaded and self.desire_model and self.awareness_model

    def _load_models(self) -> None:
        try:
            desire_path = self.model_dir / DESIRE_MODEL_FILE
            awareness_path = self.model_dir / AWARENESS_MODEL_FILE
            if not desire_path.exists() or not awareness_path.exists():
                self.logger.info("Student models missing at %s", self.model_dir)
                self._loaded = False
                return
            self.desire_model = joblib.load(desire_path)
            self.awareness_model = joblib.load(awareness_path)
            stored_threshold = self.desire_model.get("confidence_threshold")
            if stored_threshold:
                self.confidence_threshold = float(stored_threshold)
            self._loaded = True
            self.logger.info(
                "Student models loaded (trained_at=%s, samples=%s)",
                self.desire_model.get("trained_at"),
                self.desire_model.get("n_samples"),
            )
        except Exception:
            self.logger.exception("Failed to load student models from %s", self.model_dir)
            self._loaded = False

    def reload(self) -> None:
        """Force reloading the models from disk."""

        if not self.enabled:
            return
        self._load_models()

    def predict(
        self,
        sample: MappingLike,
        *,
        sig_hash: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self.is_ready:
            return None
        text = str(sample.get("text") or "").strip()
        if not text:
            return None
        try:
            desire_pred, desire_conf = self._predict_target(self.desire_model, sample)
            awareness_pred, awareness_conf = self._predict_target(self.awareness_model, sample)
        except Exception:
            self.logger.exception("Student prediction failed for %s", sig_hash)
            return None
        confidence = float(min(desire_conf, awareness_conf))
        if confidence < self.confidence_threshold:
            return None
        reason = f"Predicción alumno (conf {confidence:.0%})"
        return {
            "desire": int(round(desire_pred)),
            "awareness": int(round(awareness_pred)),
            "reason": reason[:120],
            "source": "student",
            "confidence": confidence,
        }

    def _predict_target(
        self, model_data: Dict[str, Any], sample: MappingLike
    ) -> Tuple[float, float]:
        pipeline: Pipeline = model_data["pipeline"]
        predictions = pipeline.predict([sample])
        try:
            proba = pipeline.predict_proba([sample])
            confidence = float(np.max(proba))
        except Exception:
            confidence = 0.0
        pred_value = float(predictions[0])
        return pred_value, confidence

