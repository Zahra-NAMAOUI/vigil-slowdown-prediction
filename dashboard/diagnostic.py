"""Command-line proof of raw history → exact features → preprocessing → score."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from dashboard.config import DATABASE_PATH, PROJECT_ROOT
from dashboard.inference import load_inference_engine
from dashboard.live_features import (
    RAW_CANDIDATE_FEATURES, compare_with_training_features, load_recent_run_history,
    prepare_current_feature_row,
)


def training_parity_diagnostic(train_path: Path) -> dict[str, object]:
    engine = load_inference_engine()
    columns = ["machine_id", "run_id", "segment_id", "timestamp", *RAW_CANDIDATE_FEATURES, *engine.input_feature_names]
    columns = list(dict.fromkeys(columns))
    data = pd.read_csv(train_path, usecols=columns)
    data["_timestamp_dt"] = pd.to_datetime(data["timestamp"], errors="coerce", utc=True)
    selected = None
    for _, group in data.groupby(["machine_id", "run_id", "segment_id"], sort=False):
        duration = (group["_timestamp_dt"].max() - group["_timestamp_dt"].min()).total_seconds()
        if duration >= 180 and len(group) >= 60:
            selected = group.drop(columns="_timestamp_dt").sort_values("timestamp").reset_index(drop=True)
            break
    if selected is None:
        raise RuntimeError("No sufficiently long training segment is available for parity validation.")
    parity = compare_with_training_features(selected, engine.input_feature_names)
    live_input_columns = ["machine_id", "run_id", "segment_id", "timestamp", *RAW_CANDIDATE_FEATURES]
    readiness = prepare_current_feature_row(selected[live_input_columns], engine.input_feature_names)
    if not parity["exact_match"] or not readiness.ready or readiness.feature_row is None:
        raise RuntimeError(f"Exact feature reconstruction failed: {parity}")
    result = engine.predict(readiness.feature_row)
    return {
        "source": str(train_path.relative_to(PROJECT_ROOT)),
        "machine_id": str(selected.machine_id.iloc[0]), "run_id": str(selected.run_id.iloc[0]),
        "history_rows": len(selected), "history_seconds": readiness.history_seconds,
        "feature_parity": parity, "preprocessor_input_features": len(engine.input_feature_names),
        "model_transformed_features": len(engine.transformed_feature_names), "inference": result.to_dict(),
        "threshold_unchanged": result.threshold == 0.50,
    }


def live_database_diagnostic(database_path: Path, run_id: str) -> dict[str, object]:
    engine = load_inference_engine()
    history = load_recent_run_history(database_path, run_id)
    readiness = prepare_current_feature_row(history, engine.input_feature_names)
    payload: dict[str, object] = {
        "source": str(database_path), "run_id": run_id, "history_rows": readiness.row_count,
        "history_seconds": readiness.history_seconds, "ready": readiness.ready, "reason": readiness.reason,
    }
    if readiness.ready and readiness.feature_row is not None:
        payload["inference"] = engine.predict(readiness.feature_row).to_dict()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", type=Path, default=PROJECT_ROOT / "data/modeling/train.csv")
    parser.add_argument("--live-run-id", help="optionally diagnose one active run from the live database")
    parser.add_argument("--db", type=Path, default=DATABASE_PATH)
    args = parser.parse_args()
    result = live_database_diagnostic(args.db, args.live_run_id) if args.live_run_id else training_parity_diagnostic(args.train_path)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
