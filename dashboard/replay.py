"""Replay a recorded slowdown episode through the real live pipeline.

On a server the collector measures the server: flat metrics, no episode, nothing
for the model to find. Replay mode writes recorded RAW metrics into
`system_metrics` at the collector's own cadence and stops there. Everything
downstream is untouched: live_features recomputes the 223 features from those
rows and inference scores them. No stored prediction is ever replayed.

The class mirrors CollectorController's start/stop/status surface so the API can
hold either one behind the same name.
"""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import sqlite3
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from dashboard.collector_control import CollectorControlError, CollectorStatus
from dashboard.config import PROJECT_ROOT
from dashboard.live_features import RAW_CANDIDATE_FEATURES

REPLAY_SEGMENT_PATH = PROJECT_ROOT / "data/replay/replay_segment.csv"
REPLAY_RUN_PREFIX = "replay-"
REPLAY_VERSION = "replay/1.0"

INSERT_COLUMNS = ["run_id", "machine_id", "timestamp", "elapsed_seconds", "phase",
                  *RAW_CANDIDATE_FEATURES, "sensor_errors_json", "missed_deadline"]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _ensure_collector_schema(database_path: Path) -> None:
    """Create `runs` / `system_metrics` with the collector's OWN schema definition.

    In live mode collect_agent.py creates these tables when it starts; in replay
    mode nothing else would. The schema is loaded from that file rather than
    copied here, so replay can never drift from the collector's table shape.
    collect_agent.py is imported read-only and is not modified.
    """
    collector_path = PROJECT_ROOT / "src/collect_agent.py"
    if not collector_path.exists():
        raise ReplayError(f"Collector module is missing, cannot create schema: {collector_path}")
    spec = importlib.util.spec_from_file_location("adoptai_collect_agent", collector_path)
    if spec is None or spec.loader is None:
        raise ReplayError(f"Could not load the collector module from {collector_path}")
    module = importlib.util.module_from_spec(spec)
    # Register before executing: the collector's dataclasses resolve their own
    # module through sys.modules, and would otherwise see None.
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)          # module-level code is imports and constants only
    database = module.Database(Path(database_path))
    try:
        database.initialize()
    finally:
        database.connection.close()


def _cell(value: object) -> object:
    """SQLite wants None, not NaN."""
    return None if value is None or (isinstance(value, float) and pd.isna(value)) else value


class ReplayError(RuntimeError):
    """Raised when the recorded segment cannot back a replay session."""


class ReplayCollector:
    """Plays one recorded segment into the metrics database, on a loop."""

    def __init__(
        self,
        database_path: Path,
        segment_path: Path = REPLAY_SEGMENT_PATH,
        interval_seconds: float = 2.0,
    ) -> None:
        self.database_path = Path(database_path).resolve()
        self.segment_path = Path(segment_path).resolve()
        self.interval_seconds = float(interval_seconds)

        # Read once, at startup.
        if not self.segment_path.exists():
            raise ReplayError(
                f"Replay segment is missing: {self.segment_path}. "
                "Build it with: python tools/build_replay_segment.py"
            )
        frame = pd.read_csv(self.segment_path)
        missing = sorted({"machine_id", "run_id", *RAW_CANDIDATE_FEATURES} - set(frame.columns))
        if missing:
            raise ReplayError(f"Replay segment is missing required raw columns: {missing}")
        leaked = [c for c in frame.columns if "slowdown" in c.lower()]
        if leaked:
            raise ReplayError(f"Replay segment carries label columns: {leaked}")
        if frame.empty:
            raise ReplayError("Replay segment is empty.")

        self._rows = frame[RAW_CANDIDATE_FEATURES].to_dict("records")
        self.source_run_id = str(frame["run_id"].iloc[0])
        self.source_machine_id = str(frame["machine_id"].iloc[0])
        self.row_count = len(frame)

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._run_id: str | None = None
        self._started_at: str | None = None
        self._last_sample_at: str | None = None
        self._loops = 0
        self._cursor = 0
        self._error: str | None = None

    # ── description ────────────────────────────────────────────────────────

    def source_info(self) -> dict[str, object]:
        with self._lock:
            loops, cursor, run_id = self._loops, self._cursor, self._run_id
        return {
            "source_run_id": self.source_run_id,
            "source_machine_id": self.source_machine_id,
            "segment_path": str(self.segment_path.relative_to(PROJECT_ROOT)),
            "rows": self.row_count,
            "interval_seconds": self.interval_seconds,
            "episode_seconds": round(self.row_count * self.interval_seconds, 1),
            "replay_run_id": run_id,
            "position": cursor,
            "loops_completed": loops,
        }

    # ── database ───────────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _begin_run(self, connection: sqlite3.Connection) -> None:
        """Every pass is its own run. A new run is a new segment, so the 120s
        warm-up legitimately runs again — continuity is not faked."""
        run_id = f"{REPLAY_RUN_PREFIX}{uuid.uuid4()}"
        started_at = _utc_now()
        connection.execute(
            """INSERT INTO runs (run_id, machine_id, mode, started_at_utc, status, os_name,
               os_version, architecture, hostname, python_version, collector_version,
               sample_interval_seconds, config_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (run_id, self.source_machine_id, "replay", started_at, "running",
             platform.system(), platform.version(), platform.machine(), platform.node(),
             platform.python_version(), REPLAY_VERSION, self.interval_seconds,
             json.dumps({
                 "mode": "replay",
                 "source_run_id": self.source_run_id,
                 "source_machine_id": self.source_machine_id,
                 "segment": str(self.segment_path),
                 "rows": self.row_count,
                 "note": "raw metrics replayed; features and score recomputed live",
             })),
        )
        connection.commit()
        with self._lock:
            self._run_id, self._started_at, self._cursor = run_id, started_at, 0
            self._last_sample_at = None

    def _end_run(self, connection: sqlite3.Connection, status: str) -> None:
        with self._lock:
            run_id = self._run_id
        if not run_id:
            return
        connection.execute(
            "UPDATE runs SET ended_at_utc=?, status=?, stop_reason=? WHERE run_id=?",
            (_utc_now(), status, status, run_id),
        )
        connection.commit()

    def _insert(self, connection: sqlite3.Connection, index: int) -> None:
        row = self._rows[index]
        with self._lock:
            run_id, started_at = self._run_id, self._started_at
        timestamp = _utc_now()                       # rewritten to now, on purpose
        elapsed = (
            pd.Timestamp(timestamp) - pd.Timestamp(started_at)
        ).total_seconds() if started_at else 0.0
        values = [run_id, self.source_machine_id, timestamp, round(float(elapsed), 3), "replay"]
        values += [_cell(row.get(name)) for name in RAW_CANDIDATE_FEATURES]
        values += ["{}", 0]
        placeholders = ",".join("?" * len(INSERT_COLUMNS))
        connection.execute(
            f"INSERT INTO system_metrics ({','.join(INSERT_COLUMNS)}) VALUES ({placeholders})",
            values,
        )
        connection.commit()
        with self._lock:
            self._last_sample_at = timestamp
            self._cursor = index + 1

    # ── worker ─────────────────────────────────────────────────────────────

    def _worker(self) -> None:
        try:
            _ensure_collector_schema(self.database_path)
            connection = self._connect()
        except Exception as exc:                      # surfaced through status()
            with self._lock:
                self._error = f"{type(exc).__name__}: {exc}"
            return
        try:
            self._begin_run(connection)
            index = 0
            while not self._stop_event.is_set():
                if index >= len(self._rows):
                    self._end_run(connection, "completed")
                    with self._lock:
                        self._loops += 1
                    self._begin_run(connection)
                    index = 0
                self._insert(connection, index)
                index += 1
                if self._stop_event.wait(self.interval_seconds):
                    break
            self._end_run(connection, "interrupted")
        except Exception as exc:
            with self._lock:
                self._error = f"{type(exc).__name__}: {exc}"
        finally:
            connection.close()

    # ── CollectorController-compatible surface ─────────────────────────────

    def status(self) -> CollectorStatus:
        with self._lock:
            alive = self._thread is not None and self._thread.is_alive()
            run_id, started_at, last_sample = self._run_id, self._started_at, self._last_sample_at
            loops, cursor, error = self._loops, self._cursor, self._error
        if error:
            return CollectorStatus("Error", False, True, message=f"Replay failed: {error}")
        if not alive:
            return CollectorStatus("Stopped", False, False, message="Replay is stopped.")
        if not run_id:
            return CollectorStatus("Starting", True, True, os.getpid(),
                                   message="Replay session is starting.")
        return CollectorStatus(
            "Running", True, True, os.getpid(), run_id, self.source_machine_id,
            started_at, last_sample,
            f"Replaying recorded run {self.source_run_id[:8]} — "
            f"row {cursor}/{self.row_count}, pass {loops + 1}.",
        )

    def start(self) -> CollectorStatus:
        current = self.status()
        if current.running:
            return current
        self._stop_event.clear()
        with self._lock:
            self._error = None
            self._thread = threading.Thread(target=self._worker, name="vigil-replay", daemon=True)
            self._thread.start()
        for _ in range(30):                      # let the first run row appear
            result = self.status()
            if result.state == "Running":
                return result
            self._stop_event.wait(0.1)
        return self.status()

    def stop(self, timeout_seconds: float = 12.0) -> CollectorStatus:
        with self._lock:
            thread = self._thread
        if thread is None or not thread.is_alive():
            with self._lock:
                self._thread = None
            return CollectorStatus("Stopped", False, False, message="Replay is already stopped.")
        self._stop_event.set()
        thread.join(timeout=timeout_seconds)
        if thread.is_alive():
            raise CollectorControlError("Replay did not stop within the safety timeout.")
        with self._lock:
            self._thread = None
            self._run_id = None
        return CollectorStatus("Stopped", False, False,
                               message="Replay stopped; recorded rows were preserved.")
