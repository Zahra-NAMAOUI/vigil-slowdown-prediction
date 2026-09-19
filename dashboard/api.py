"""FastAPI transport layer over the frozen AdoptAI V1 inference chain.

This module holds no business logic. It calls the same functions, in the same
order, with the same error handling as dashboard/app.py, and serves the result
as JSON. The frozen modules (config, live_features, inference, history,
collector_control) and every artifact under models/ are imported read-only.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from dashboard.collector_control import CollectorControlError, CollectorController
from dashboard.config import (
    DATABASE_PATH, FINAL_TEST_METRICS_PATH, HISTORY_MINUTES, MODEL_METADATA_PATH,
    MODEL_PATH, PREPROCESSOR_PATH, PROJECT_ROOT, REFERENCE_THRESHOLD, WARMUP_SECONDS,
)
from dashboard.history import (
    initialize_dashboard_tables, load_alerts, load_metrics_history,
    load_prediction_history, log_prediction, session_sample_count,
)
from dashboard.inference import InferenceError, load_inference_engine
from dashboard.live_features import (
    DYNAMIC_METRICS, LiveFeatureError, load_recent_run_history, prepare_current_feature_row,
)
from dashboard.replay import ReplayCollector

API_VERSION = "1.0.0"
WEB_DIR = Path(__file__).resolve().parent / "web"
FINAL_TEST_BY_MACHINE_PATH = PROJECT_ROOT / "reports/final_test_by_machine.csv"

SAMPLE_COLUMNS = [
    "cpu_pct", "ram_pct", "swap_pct", "disk_usage_pct",
    "disk_latency_ms", "context_switches_per_s", "process_count", "thread_count",
]
# Plain-language labels for the provisional driver panel.
METRIC_LABELS: dict[str, tuple[str, str, int]] = {
    "cpu_pct": ("CPU usage", "%", 1),
    "ram_pct": ("RAM usage", "%", 1),
    "swap_pct": ("Swap usage", "%", 1),
    "disk_latency_ms": ("Disk latency", "ms", 2),
    "disk_read_mb_s": ("Disk read", "MB/s", 2),
    "disk_write_mb_s": ("Disk write", "MB/s", 2),
    "net_sent_mb_s": ("Network sent", "MB/s", 2),
    "net_recv_mb_s": ("Network received", "MB/s", 2),
    "network_latency_ms": ("Network latency", "ms", 1),
    "context_switches_per_s": ("Context switches", "/s", 0),
    "process_count": ("Processes", "", 0),
    "thread_count": ("Threads", "", 0),
}

app = FastAPI(title="Vigil — AdoptAI V1", version=API_VERSION)
initialize_dashboard_tables(DATABASE_PATH)

# ─────────────────────────────────────────────────────────────────────────────
# The single mode selection point. Everything downstream — every endpoint, the
# feature engineering, the model — is identical in both modes and never asks
# which one is active. Replay only changes where the raw rows come from.
# ─────────────────────────────────────────────────────────────────────────────
VIGIL_MODE = os.environ.get("VIGIL_MODE", "live").strip().lower()
if VIGIL_MODE not in {"live", "replay"}:
    VIGIL_MODE = "live"

if VIGIL_MODE == "replay":
    controller: object = ReplayCollector(DATABASE_PATH)
    mode_detail = controller.source_info
    controller.start()          # a visitor must find the episode already playing
else:
    controller = CollectorController(DATABASE_PATH)
    mode_detail = lambda: None  # noqa: E731 — keeps the branch to this one block


def _number(value: Any) -> float | None:
    """JSON-safe float: NaN and infinity become null rather than invalid JSON."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) or math.isinf(number) else number


def _engine():
    try:
        return load_inference_engine()
    except (InferenceError, OSError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=f"Frozen model unavailable: {exc}") from exc


def _records(frame: pd.DataFrame, timestamp_column: str | None = None) -> list[dict[str, Any]]:
    """Rows as JSON-safe dicts. Only `timestamp_column` is parsed as a datetime."""
    if frame.empty:
        return []
    rows: list[dict[str, Any]] = []
    for record in frame.to_dict("records"):
        clean: dict[str, Any] = {}
        for key, value in record.items():
            if key == timestamp_column:
                moment = pd.to_datetime(value, errors="coerce", utc=True)
                clean[key] = None if pd.isna(moment) else moment.isoformat()
            elif isinstance(value, str) or value is None:
                clean[key] = value
            else:
                clean[key] = _number(value)
        rows.append(clean)
    return rows


def _collector_payload(status, sample_count: int) -> dict[str, Any]:
    payload = status.to_dict()
    payload["sample_count"] = sample_count
    return payload


def _score_drivers(feature_row: pd.DataFrame, limit: int = 5) -> list[dict[str, Any]]:
    """Provisional explanation: raw metrics furthest from their own 120s baseline.

    ### SHAP REPLACEMENT POINT ###
    This ranks each dynamic metric by (value - rolling mean) / rolling std, read
    straight off the engineered feature row. It is a descriptive signal, not the
    model's own attribution. When SHAP lands, replace this function with one that
    returns per-feature SHAP values in the same {key,label,value,unit,digits,z}
    shape; the front end renderer needs no change.
    """
    row = feature_row.iloc[0]
    drivers: list[dict[str, Any]] = []
    for metric in DYNAMIC_METRICS:
        value = _number(row.get(metric))
        baseline = _number(row.get(f"{metric}_mean_120s"))
        spread = _number(row.get(f"{metric}_std_120s"))
        if value is None or baseline is None or not spread:
            continue
        label, unit, digits = METRIC_LABELS[metric]
        drivers.append({
            "key": metric, "label": label, "unit": unit, "digits": digits,
            "value": value, "baseline": baseline, "z": (value - baseline) / spread,
        })
    drivers.sort(key=lambda driver: abs(driver["z"]), reverse=True)
    return drivers[:limit]


def _active_run_id(status) -> str | None:
    """The running session, or the most recent one so history survives a stop."""
    return status.run_id or session_sample_count(DATABASE_PATH)[0]


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/live")
def live() -> dict[str, Any]:
    engine = _engine()
    status = controller.status()
    _, sample_count = session_sample_count(DATABASE_PATH, status.run_id)
    payload: dict[str, Any] = {
        "ready": False, "elapsed": 0.0, "required": WARMUP_SECONDS, "progress": 0.0,
        "reason": "", "collector": _collector_payload(status, sample_count),
        "sample": None, "score": None, "drivers": [],
    }
    if not status.run_id:
        payload["reason"] = "No collection session is running."
        return payload

    try:
        raw_history = load_recent_run_history(DATABASE_PATH, status.run_id)
        if raw_history.empty:
            payload["reason"] = "Collector started. Waiting for the first stored measurement."
            return payload
        readiness = prepare_current_feature_row(raw_history, engine.input_feature_names)
        latest = raw_history.iloc[-1]
        result = (
            engine.predict(readiness.feature_row)
            if readiness.ready and readiness.feature_row is not None
            else None
        )
        if result is not None:
            log_prediction(
                DATABASE_PATH, str(latest.timestamp), str(latest.machine_id), str(latest.run_id),
                result, _number(latest.cpu_pct), _number(latest.ram_pct),
            )
    except (LiveFeatureError, InferenceError, OSError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=f"Live inference is unavailable: {exc}") from exc

    payload.update(
        ready=readiness.ready,
        elapsed=round(readiness.history_seconds, 1),
        progress=readiness.progress,
        reason=readiness.reason,
        gap_threshold_seconds=readiness.gap_threshold_seconds,
        sample={
            "timestamp": str(latest.timestamp),
            **{column: _number(latest.get(column)) for column in SAMPLE_COLUMNS},
        },
    )
    if result is not None:
        payload["score"] = result.to_dict()
        payload["drivers"] = _score_drivers(readiness.feature_row)
    return payload


@app.get("/api/history")
def history(minutes: int = Query(HISTORY_MINUTES, ge=1, le=180)) -> dict[str, Any]:
    run_id = _active_run_id(controller.status())
    if not run_id:
        return {"run_id": None, "minutes": minutes, "metrics": [], "predictions": []}

    metrics = load_metrics_history(DATABASE_PATH, run_id, limit=1_000)
    predictions = load_prediction_history(DATABASE_PATH, run_id, limit=1_000)
    if not metrics.empty:
        cutoff = metrics.timestamp.max() - pd.Timedelta(minutes=minutes)
        metrics = metrics.loc[metrics.timestamp.ge(cutoff)]
        if not predictions.empty:
            predictions = predictions.loc[predictions.timestamp_utc.ge(cutoff)]
    return {
        "run_id": run_id,
        "minutes": minutes,
        "metrics": _records(metrics, "timestamp"),
        "predictions": _records(predictions, "timestamp_utc"),
    }


@app.get("/api/alerts")
def alerts(limit: int = Query(12, ge=1, le=100)) -> dict[str, Any]:
    run_id = _active_run_id(controller.status())
    if not run_id:
        return {"run_id": None, "alerts": []}
    return {"run_id": run_id, "alerts": _records(load_alerts(DATABASE_PATH, run_id, limit), "timestamp_utc")}


@app.get("/api/model")
def model() -> dict[str, Any]:
    try:
        metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
        test_metrics = pd.read_csv(FINAL_TEST_METRICS_PATH)
        by_machine = pd.read_csv(FINAL_TEST_BY_MACHINE_PATH)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=f"Model reports unavailable: {exc}") from exc
    return {
        "metadata": metadata,
        "test_metrics": _records(test_metrics)[0] if not test_metrics.empty else {},
        "by_machine": _records(by_machine),
    }


@app.get("/api/collector/status")
def collector_status() -> dict[str, Any]:
    status = controller.status()
    return _collector_payload(status, session_sample_count(DATABASE_PATH, status.run_id)[1])


@app.post("/api/collector/start")
def collector_start() -> dict[str, Any]:
    if controller.status().running:
        raise HTTPException(status_code=409, detail="A collector is already running.")
    try:
        status = controller.start()
    except (CollectorControlError, OSError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _collector_payload(status, session_sample_count(DATABASE_PATH, status.run_id)[1])


@app.post("/api/collector/stop")
def collector_stop() -> dict[str, Any]:
    try:
        status = controller.stop()
    except CollectorControlError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _collector_payload(status, session_sample_count(DATABASE_PATH, status.run_id)[1])


@app.get("/api/health")
def health() -> dict[str, Any]:
    model_ready, model_error = True, None
    try:
        load_inference_engine()
    except (InferenceError, OSError, ValueError) as exc:
        model_ready, model_error = False, str(exc)
    return {
        "status": "ok" if model_ready and DATABASE_PATH.exists() else "degraded",
        "mode": VIGIL_MODE,
        "replay": mode_detail(),
        "model_ready": model_ready,
        "model_error": model_error,
        "database_reachable": DATABASE_PATH.exists(),
        "database_path": str(DATABASE_PATH),
        "model_path": str(MODEL_PATH),
        "preprocessor_path": str(PREPROCESSOR_PATH),
        "decision_threshold": REFERENCE_THRESHOLD,
        "warmup_seconds": WARMUP_SECONDS,
        "api_version": API_VERSION,
    }


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
