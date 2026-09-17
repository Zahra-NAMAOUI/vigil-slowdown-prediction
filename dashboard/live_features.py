"""Exact causal feature engineering for the frozen AdoptAI V1 model.

The formulas in this module are extracted from notebook 08.  They intentionally
preserve its column names, time-window semantics, ddof, missingness calculation,
and machine/run/segment boundaries.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from dashboard.config import MAX_RECENT_ROWS, WARMUP_SECONDS


RAW_CANDIDATE_FEATURES = [
    "cpu_pct", "cpu_frequency_mhz", "ram_pct", "ram_used_mb", "ram_available_mb",
    "swap_pct", "swap_used_mb", "disk_usage_pct", "disk_free_gb", "disk_read_mb_s",
    "disk_write_mb_s", "disk_latency_ms", "net_sent_mb_s", "net_recv_mb_s",
    "network_latency_ms", "process_count", "thread_count", "context_switches_per_s",
    "battery_pct", "battery_plugged",
]
DYNAMIC_METRICS = [
    "cpu_pct", "ram_pct", "swap_pct", "disk_latency_ms", "disk_read_mb_s",
    "disk_write_mb_s", "net_sent_mb_s", "net_recv_mb_s", "network_latency_ms",
    "context_switches_per_s", "process_count", "thread_count",
]
ROLLING_WINDOWS_SECONDS = [30, 60, 120]
ROLLING_STATISTICS = ["mean", "max", "min", "std", "change", "missing_pct"]
DIFFERENCE_METRICS = ["cpu_pct", "ram_pct", "swap_pct", "disk_latency_ms", "context_switches_per_s"]
MISSING_INDICATOR_METRICS = ["context_switches_per_s", "network_latency_ms"]
SEQUENCE_KEYS = ["machine_id", "run_id", "segment_id"]
PROHIBITED_MODEL_COLUMNS = {
    "slowdown_now", "slowdown_in_5min", "slowdown_in_10min", "valid_5min_horizon",
    "valid_10min_horizon", "status", "ended_at_utc", "run_complete", "missed_deadline",
    "id", "machine_id", "run_id", "segment_id", "timestamp", "elapsed_seconds", "phase",
    "legacy_id", "stress_cpu_target_pct", "stress_memory_target_mb", "temperature_c",
    "gpu_usage_pct", "cpu_per_core_json", "gpu_per_device_json", "sensor_errors_json",
    "sample_reliable",
}
LIVE_QUERY_COLUMNS = [
    "id", "run_id", "machine_id", "timestamp", "elapsed_seconds", "phase",
    *RAW_CANDIDATE_FEATURES, "sensor_errors_json", "missed_deadline",
]


class LiveFeatureError(RuntimeError):
    """Raised when live data cannot safely satisfy the frozen feature contract."""


@dataclass(frozen=True)
class FeatureReadiness:
    ready: bool
    progress: float
    history_seconds: float
    row_count: int
    segment_id: str | int | None
    gap_threshold_seconds: float
    reason: str
    feature_row: pd.DataFrame | None = None


def generated_feature_names() -> list[str]:
    rolling = [
        f"{metric}_{statistic}_{seconds}s"
        for metric in DYNAMIC_METRICS
        for seconds in ROLLING_WINDOWS_SECONDS
        for statistic in ROLLING_STATISTICS
    ]
    differences = [f"{metric}_diff" for metric in DIFFERENCE_METRICS]
    missing = [f"{metric}_missing" for metric in MISSING_INDICATOR_METRICS]
    return rolling + differences + missing


MODEL_CANDIDATE_FEATURES = RAW_CANDIDATE_FEATURES + generated_feature_names()


def validate_raw_schema(frame: pd.DataFrame) -> None:
    required = {"machine_id", "run_id", "timestamp", *RAW_CANDIDATE_FEATURES}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise LiveFeatureError(f"Live collector schema is missing required columns: {missing}")
    if frame.empty:
        raise LiveFeatureError("No measurements are available for the active run yet.")


def load_recent_run_history(database_path: Path, run_id: str, limit: int = MAX_RECENT_ROWS) -> pd.DataFrame:
    """Read only the most recent rows for one run; never scan the full database."""
    if not database_path.exists():
        raise LiveFeatureError(f"Metrics database does not exist: {database_path}")
    columns = ",".join(f'"{column}"' for column in LIVE_QUERY_COLUMNS)
    query = f"""
        SELECT {columns}
        FROM system_metrics
        WHERE run_id = ?
        ORDER BY timestamp DESC, id DESC
        LIMIT ?
    """
    with sqlite3.connect(database_path, timeout=5) as connection:
        frame = pd.read_sql_query(query, connection, params=(run_id, int(limit)))
        run_row = connection.execute(
            "SELECT sample_interval_seconds FROM runs WHERE run_id=?", (run_id,)
        ).fetchone()
    if frame.empty:
        return frame
    frame = frame.iloc[::-1].reset_index(drop=True)
    frame.attrs["expected_interval_seconds"] = float(run_row[0]) if run_row else 2.0
    return frame


def assign_live_segments(frame: pd.DataFrame, expected_interval_seconds: float = 2.0) -> tuple[pd.DataFrame, float]:
    """Apply the training gap rule causally and return a segment ID for each live row.

    Training used max(5 × run median interval, 10 seconds). During an active run,
    the configured collector interval is the stable reference. The observed median
    is used once three positive deltas exist. Any detected gap resets warm-up, so a
    live feature window never bridges an interruption.
    """
    validate_raw_schema(frame)
    result = frame.copy(deep=True)
    result["_timestamp_dt"] = pd.to_datetime(result["timestamp"], errors="coerce", utc=True)
    if result["_timestamp_dt"].isna().any():
        raise LiveFeatureError("Invalid timestamps prevent safe live rolling windows.")
    if result[["machine_id", "run_id"]].drop_duplicates().shape[0] != 1:
        raise LiveFeatureError("Live history must contain exactly one machine/run pair.")
    result = result.sort_values(["_timestamp_dt", "id"] if "id" in result else ["_timestamp_dt"], kind="mergesort")
    if result.duplicated(["machine_id", "run_id", "_timestamp_dt"]).any():
        raise LiveFeatureError("Duplicate timestamps make live causal ordering ambiguous.")
    deltas = result["_timestamp_dt"].diff().dt.total_seconds()
    positive = deltas[deltas.gt(0)]
    normal_interval = float(positive.median()) if len(positive) >= 3 else float(expected_interval_seconds)
    gap_threshold = max(5.0 * normal_interval, 10.0)
    segment_number = deltas.gt(gap_threshold).fillna(False).cumsum().astype(int)
    run_id = str(result["run_id"].iloc[0])
    result["segment_id"] = segment_number.map(lambda number: f"{run_id}:live:{number}")
    return result.reset_index(drop=True), float(gap_threshold)


def engineer_feature_history(raw_history: pd.DataFrame) -> pd.DataFrame:
    """Recreate notebook 08 features exactly for supplied bounded history."""
    validate_raw_schema(raw_history)
    if "segment_id" not in raw_history.columns:
        raise LiveFeatureError("segment_id is required before feature engineering.")
    feature_work = raw_history.copy(deep=True)
    feature_work["_source_row"] = np.arange(len(feature_work))
    feature_work["_timestamp_dt"] = pd.to_datetime(feature_work["timestamp"], errors="coerce", utc=True)
    if feature_work["_timestamp_dt"].isna().any():
        raise LiveFeatureError("Invalid timestamps prevent safe rolling windows.")
    if feature_work.duplicated(SEQUENCE_KEYS + ["_timestamp_dt"]).any():
        raise LiveFeatureError("Duplicated within-segment timestamps make causal ordering ambiguous.")
    feature_work = feature_work.sort_values(
        SEQUENCE_KEYS + ["_timestamp_dt", "_source_row"], kind="mergesort"
    ).reset_index(drop=True)

    generated_arrays: dict[str, np.ndarray] = {}
    row_count = len(feature_work)
    for metric in DYNAMIC_METRICS:
        for seconds in ROLLING_WINDOWS_SECONDS:
            for statistic in ROLLING_STATISTICS:
                generated_arrays[f"{metric}_{statistic}_{seconds}s"] = np.full(row_count, np.nan, dtype=float)

    for _, segment in feature_work.groupby(SEQUENCE_KEYS, sort=False):
        ordered = segment.sort_values("_timestamp_dt")
        positions = ordered.index.to_numpy()
        indexed = ordered.set_index("_timestamp_dt")
        for seconds in ROLLING_WINDOWS_SECONDS:
            window = f"{seconds}s"
            total = pd.Series(1.0, index=indexed.index).rolling(window, min_periods=1, closed="right").sum()
            for metric in DYNAMIC_METRICS:
                rolling = indexed[metric].rolling(window, min_periods=1, closed="right")
                generated_arrays[f"{metric}_mean_{seconds}s"][positions] = rolling.mean().to_numpy()
                generated_arrays[f"{metric}_max_{seconds}s"][positions] = rolling.max().to_numpy()
                generated_arrays[f"{metric}_min_{seconds}s"][positions] = rolling.min().to_numpy()
                generated_arrays[f"{metric}_std_{seconds}s"][positions] = rolling.std(ddof=1).to_numpy()
                generated_arrays[f"{metric}_change_{seconds}s"][positions] = rolling.apply(
                    lambda values: values.iloc[-1] - values.iloc[0]
                    if pd.notna(values.iloc[-1]) and pd.notna(values.iloc[0]) else np.nan,
                    raw=False,
                ).to_numpy()
                generated_arrays[f"{metric}_missing_pct_{seconds}s"][positions] = (
                    (total - rolling.count()) / total * 100
                ).to_numpy()

    feature_work = pd.concat([feature_work, pd.DataFrame(generated_arrays)], axis=1)
    for metric in DIFFERENCE_METRICS:
        feature_work[f"{metric}_diff"] = feature_work.groupby(SEQUENCE_KEYS, sort=False)[metric].diff()
    for metric in MISSING_INDICATOR_METRICS:
        feature_work[f"{metric}_missing"] = feature_work[metric].isna().astype("int8")
    return feature_work


def prepare_current_feature_row(
    raw_history: pd.DataFrame,
    expected_feature_names: Sequence[str],
    expected_interval_seconds: float | None = None,
    warmup_seconds: float = WARMUP_SECONDS,
) -> FeatureReadiness:
    """Return the exact current feature row only after continuous warm-up."""
    if raw_history.empty:
        return FeatureReadiness(False, 0.0, 0.0, 0, None, 10.0, "Waiting for the first measurement.")
    interval = expected_interval_seconds or float(raw_history.attrs.get("expected_interval_seconds", 2.0))
    if "segment_id" in raw_history.columns:
        segmented = raw_history.copy(deep=True)
        segmented["_timestamp_dt"] = pd.to_datetime(segmented["timestamp"], errors="coerce", utc=True)
        gap_threshold = max(5.0 * interval, 10.0)
    else:
        segmented, gap_threshold = assign_live_segments(raw_history, interval)
    current_segment_id = segmented["segment_id"].iloc[-1]
    current = segmented.loc[segmented["segment_id"].eq(current_segment_id)].copy()
    current["_timestamp_dt"] = pd.to_datetime(current["timestamp"], errors="coerce", utc=True)
    history_seconds = float((current["_timestamp_dt"].max() - current["_timestamp_dt"].min()).total_seconds())
    progress = min(1.0, max(0.0, history_seconds / warmup_seconds))
    if history_seconds < warmup_seconds:
        return FeatureReadiness(
            False, progress, history_seconds, len(current), current_segment_id, gap_threshold,
            f"Collecting continuous history ({history_seconds:.0f}/{warmup_seconds:.0f} seconds).",
        )
    engineered = engineer_feature_history(current)
    expected = list(expected_feature_names)
    missing = sorted(set(expected) - set(engineered.columns))
    if missing:
        raise LiveFeatureError(f"Exact live reconstruction is missing model inputs: {missing}")
    if set(expected) & PROHIBITED_MODEL_COLUMNS or any("slowdown" in column.lower() for column in expected):
        raise LiveFeatureError("Target leakage was detected in the frozen feature contract.")
    feature_row = engineered.iloc[[-1]][expected].copy()
    if feature_row.columns.tolist() != expected or feature_row.shape != (1, len(expected)):
        raise LiveFeatureError("Engineered feature order does not match the frozen preprocessing contract.")
    numeric = feature_row.to_numpy(dtype=float)
    if np.isinf(numeric).any():
        raise LiveFeatureError("Engineered live features contain infinite values.")
    return FeatureReadiness(
        True, 1.0, history_seconds, len(current), current_segment_id, gap_threshold,
        "Continuous history is sufficient for inference.", feature_row,
    )


def compare_with_training_features(
    raw_and_engineered_history: pd.DataFrame,
    expected_feature_names: Iterable[str],
) -> dict[str, float | int | bool]:
    """Prove extracted formulas reproduce an already-engineered training row."""
    expected = list(expected_feature_names)
    required = ["machine_id", "run_id", "segment_id", "timestamp", *RAW_CANDIDATE_FEATURES]
    rebuilt = engineer_feature_history(raw_and_engineered_history[required])
    actual = raw_and_engineered_history.iloc[-1][expected].to_numpy(dtype=float)
    reproduced = rebuilt.iloc[-1][expected].to_numpy(dtype=float)
    equal = np.isclose(actual, reproduced, rtol=1e-9, atol=1e-9, equal_nan=True)
    finite_diff = np.abs(actual[equal == False] - reproduced[equal == False])
    finite_diff = finite_diff[np.isfinite(finite_diff)]
    return {
        "feature_count": len(expected),
        "matching_features": int(equal.sum()),
        "mismatching_features": int((~equal).sum()),
        "max_absolute_error": float(finite_diff.max()) if len(finite_diff) else 0.0,
        "exact_match": bool(equal.all()),
    }
