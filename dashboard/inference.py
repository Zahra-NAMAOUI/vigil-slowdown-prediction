"""Frozen LightGBM V1 inference with strict artifact and schema checks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from dashboard.config import MODEL_METADATA_PATH, MODEL_PATH, PREPROCESSOR_PATH, REFERENCE_THRESHOLD


class InferenceError(RuntimeError):
    """Raised when frozen artifacts or a live feature row are incompatible."""


@dataclass(frozen=True)
class InferenceResult:
    model_score: float
    risk_score: float
    predicted_class: int
    status: str
    threshold: float = REFERENCE_THRESHOLD
    horizon: str = "next 5 minutes"
    score_is_calibrated_probability: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def risk_status(score: float) -> str:
    if score < 0.30:
        return "Low"
    if score < 0.50:
        return "Moderate"
    if score < 0.70:
        return "Warning"
    return "High Risk"


class FrozenInferenceEngine:
    """Load immutable artifacts once and serve validated one-row inference."""

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        preprocessor_path: Path = PREPROCESSOR_PATH,
        metadata_path: Path = MODEL_METADATA_PATH,
    ) -> None:
        for path in (model_path, preprocessor_path, metadata_path):
            if not path.exists():
                raise InferenceError(f"Required frozen artifact is missing: {path}")
        self.model_path = model_path.resolve()
        self.preprocessor_path = preprocessor_path.resolve()
        self.metadata_path = metadata_path.resolve()
        self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        expected_model_hash = self.metadata.get("artifacts", {}).get("model_sha256")
        expected_preprocessor_hash = self.metadata.get("artifacts", {}).get("preprocessor_sha256")
        if expected_model_hash and _sha256(self.model_path) != expected_model_hash:
            raise InferenceError("Baseline model checksum does not match final_v1 metadata.")
        if expected_preprocessor_hash and _sha256(self.preprocessor_path) != expected_preprocessor_hash:
            raise InferenceError("Tree preprocessor checksum does not match final_v1 metadata.")
        self.model = joblib.load(self.model_path)
        self.preprocessor = joblib.load(self.preprocessor_path)
        self.input_feature_names = list(self.preprocessor.feature_names_in_)
        imputer = self.preprocessor.named_steps["imputer"]
        indicator_sources = [self.input_feature_names[index] for index in imputer.indicator_.features_]
        self.transformed_feature_names = self.input_feature_names + [
            f"{name}__missing_indicator" for name in indicator_sources
        ]
        model_names = list(self.model.booster_.feature_name())
        if self.transformed_feature_names != model_names:
            raise InferenceError("Preprocessor output order does not match the frozen LightGBM feature order.")
        if len(self.input_feature_names) != 228 or len(self.transformed_feature_names) != 360:
            raise InferenceError("Frozen feature counts do not match the current 153k V1 experiment.")
        if getattr(self.model, "n_features_in_", 360) != 360:
            raise InferenceError("Frozen model expects an unexpected transformed feature count.")

    def predict(self, feature_row: pd.DataFrame) -> InferenceResult:
        if not isinstance(feature_row, pd.DataFrame) or len(feature_row) != 1:
            raise InferenceError("Inference requires exactly one engineered feature row.")
        if feature_row.columns.tolist() != self.input_feature_names:
            raise InferenceError("Live feature names/order do not match the frozen preprocessing contract.")
        if any("slowdown" in name.lower() for name in feature_row.columns):
            raise InferenceError("Target leakage detected in the inference row.")
        values = feature_row.to_numpy(dtype=float)
        if np.isinf(values).any():
            raise InferenceError("Live input contains infinite values.")
        transformed = self.preprocessor.transform(feature_row)
        if transformed.shape != (1, 360) or not np.isfinite(transformed).all():
            raise InferenceError("Frozen preprocessing did not produce one finite 360-feature row.")
        model_score = float(self.model.predict_proba(transformed)[:, 1][0])
        if not np.isfinite(model_score) or not 0.0 <= model_score <= 1.0:
            raise InferenceError("Frozen model produced an invalid score.")
        return InferenceResult(
            model_score=model_score,
            risk_score=100.0 * model_score,
            predicted_class=int(model_score >= REFERENCE_THRESHOLD),
            status=risk_status(model_score),
        )


@lru_cache(maxsize=4)
def load_inference_engine(
    model_path: str = str(MODEL_PATH),
    preprocessor_path: str = str(PREPROCESSOR_PATH),
    metadata_path: str = str(MODEL_METADATA_PATH),
) -> FrozenInferenceEngine:
    """Cache frozen artifacts for the process lifetime (including Streamlit reruns)."""
    return FrozenInferenceEngine(Path(model_path), Path(preprocessor_path), Path(metadata_path))

