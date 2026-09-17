"""Bounded SQLite reads and lightweight prediction/alert persistence."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from dashboard.config import ALERT_COOLDOWN_SECONDS
from dashboard.inference import InferenceResult


PREDICTION_TABLE = "dashboard_predictions"
ALERT_TABLE = "dashboard_alerts"


def connect(database_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(database_path, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA journal_mode=WAL")
    return connection


def initialize_dashboard_tables(database_path: Path) -> None:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with connect(database_path) as connection:
        connection.executescript(
            f"""
            CREATE TABLE IF NOT EXISTS {PREDICTION_TABLE} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_utc TEXT NOT NULL,
                machine_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                model_score REAL NOT NULL,
                risk_score REAL NOT NULL,
                predicted_class INTEGER NOT NULL,
                status TEXT NOT NULL,
                cpu_pct REAL,
                ram_pct REAL,
                UNIQUE(run_id, timestamp_utc)
            );
            CREATE INDEX IF NOT EXISTS idx_dashboard_predictions_run_ts
                ON {PREDICTION_TABLE}(run_id, timestamp_utc);
            CREATE TABLE IF NOT EXISTS {ALERT_TABLE} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER NOT NULL REFERENCES {PREDICTION_TABLE}(id),
                timestamp_utc TEXT NOT NULL,
                machine_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                risk_score REAL NOT NULL,
                status TEXT NOT NULL,
                cpu_pct REAL,
                ram_pct REAL,
                alert_reason TEXT NOT NULL,
                UNIQUE(prediction_id)
            );
            CREATE INDEX IF NOT EXISTS idx_dashboard_alerts_run_ts
                ON {ALERT_TABLE}(run_id, timestamp_utc);
            """
        )


def log_prediction(
    database_path: Path,
    timestamp_utc: str,
    machine_id: str,
    run_id: str,
    result: InferenceResult,
    cpu_pct: float | None,
    ram_pct: float | None,
    cooldown_seconds: int = ALERT_COOLDOWN_SECONDS,
) -> tuple[bool, bool]:
    """Insert once per sample and create a deduplicated risk alert when appropriate."""
    initialize_dashboard_tables(database_path)
    with connect(database_path) as connection:
        previous = connection.execute(
            f"SELECT predicted_class FROM {PREDICTION_TABLE} WHERE run_id=? ORDER BY timestamp_utc DESC,id DESC LIMIT 1",
            (run_id,),
        ).fetchone()
        cursor = connection.execute(
            f"""INSERT OR IGNORE INTO {PREDICTION_TABLE}
            (timestamp_utc,machine_id,run_id,model_score,risk_score,predicted_class,status,cpu_pct,ram_pct)
            VALUES(?,?,?,?,?,?,?,?,?)""",
            (timestamp_utc, machine_id, run_id, result.model_score, result.risk_score,
             result.predicted_class, result.status, cpu_pct, ram_pct),
        )
        inserted = cursor.rowcount == 1
        if not inserted or result.predicted_class != 1:
            return inserted, False
        prediction_id = int(cursor.lastrowid)
        last_alert = connection.execute(
            f"SELECT timestamp_utc FROM {ALERT_TABLE} WHERE run_id=? ORDER BY timestamp_utc DESC,id DESC LIMIT 1",
            (run_id,),
        ).fetchone()
        transitioned = previous is None or int(previous[0]) != 1
        cooldown_elapsed = True
        if last_alert:
            now = pd.to_datetime(timestamp_utc, utc=True)
            then = pd.to_datetime(last_alert[0], utc=True)
            cooldown_elapsed = (now - then).total_seconds() >= cooldown_seconds
        if not (transitioned or cooldown_elapsed):
            return True, False
        reason = "entered_risk_state" if transitioned else "risk_state_cooldown_refresh"
        connection.execute(
            f"""INSERT INTO {ALERT_TABLE}
            (prediction_id,timestamp_utc,machine_id,run_id,risk_score,status,cpu_pct,ram_pct,alert_reason)
            VALUES(?,?,?,?,?,?,?,?,?)""",
            (prediction_id, timestamp_utc, machine_id, run_id, result.risk_score,
             result.status, cpu_pct, ram_pct, reason),
        )
        return True, True


def list_runs(database_path: Path, limit: int = 50) -> pd.DataFrame:
    if not database_path.exists():
        return pd.DataFrame()
    initialize_dashboard_tables(database_path)
    query = f"""
        SELECT r.run_id,r.machine_id,r.mode,r.started_at_utc,r.ended_at_utc,r.status,
               r.sample_interval_seconds,COUNT(DISTINCT m.id) AS measurement_rows,
               COUNT(DISTINCT p.id) AS prediction_rows
        FROM runs r
        LEFT JOIN system_metrics m ON m.run_id=r.run_id
        LEFT JOIN {PREDICTION_TABLE} p ON p.run_id=r.run_id
        GROUP BY r.run_id
        ORDER BY r.started_at_utc DESC
        LIMIT ?
    """
    with connect(database_path) as connection:
        has_collector_schema = connection.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name IN ('runs','system_metrics')"
        ).fetchone()[0] == 2
        if not has_collector_schema:
            return pd.DataFrame()
        return pd.read_sql_query(query, connection, params=(int(limit),))


def session_sample_count(database_path: Path, run_id: str | None = None) -> tuple[str | None, int]:
    """Return a bounded active-run count, or the latest session count when stopped."""
    if not database_path.exists():
        return None, 0
    with connect(database_path) as connection:
        has_collector_schema = connection.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name IN ('runs','system_metrics')"
        ).fetchone()[0] == 2
        if not has_collector_schema:
            return None, 0
        if run_id:
            row = connection.execute(
                "SELECT ?, COUNT(*) FROM system_metrics WHERE run_id=?", (run_id, run_id)
            ).fetchone()
        else:
            row = connection.execute(
                """
                SELECT r.run_id, (SELECT COUNT(*) FROM system_metrics m WHERE m.run_id=r.run_id)
                FROM runs r ORDER BY r.started_at_utc DESC LIMIT 1
                """
            ).fetchone()
    return (str(row[0]), int(row[1])) if row else (None, 0)


def load_metrics_history(database_path: Path, run_id: str, limit: int = 10_000) -> pd.DataFrame:
    columns = "timestamp,cpu_pct,ram_pct,swap_pct,disk_usage_pct,disk_latency_ms,context_switches_per_s,process_count,thread_count"
    query = f"SELECT {columns} FROM system_metrics WHERE run_id=? ORDER BY timestamp DESC,id DESC LIMIT ?"
    with connect(database_path) as connection:
        has_metrics = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='system_metrics'"
        ).fetchone()
        if not has_metrics:
            return pd.DataFrame()
        frame = pd.read_sql_query(query, connection, params=(run_id, int(limit)))
    if not frame.empty:
        frame = frame.iloc[::-1].reset_index(drop=True)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    return frame


def load_prediction_history(database_path: Path, run_id: str, limit: int = 10_000) -> pd.DataFrame:
    initialize_dashboard_tables(database_path)
    query = f"SELECT * FROM {PREDICTION_TABLE} WHERE run_id=? ORDER BY timestamp_utc DESC,id DESC LIMIT ?"
    with connect(database_path) as connection:
        frame = pd.read_sql_query(query, connection, params=(run_id, int(limit)))
    if not frame.empty:
        frame = frame.iloc[::-1].reset_index(drop=True)
        frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], errors="coerce", utc=True)
    return frame


def load_alerts(database_path: Path, run_id: str | None = None, limit: int = 25) -> pd.DataFrame:
    initialize_dashboard_tables(database_path)
    where = "WHERE run_id=?" if run_id else ""
    params = (run_id, int(limit)) if run_id else (int(limit),)
    query = f"SELECT * FROM {ALERT_TABLE} {where} ORDER BY timestamp_utc DESC,id DESC LIMIT ?"
    with connect(database_path) as connection:
        return pd.read_sql_query(query, connection, params=params)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
