"""Safe lifecycle control for the existing AdoptAI collector subprocess."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import psutil

from dashboard.config import (
    COLLECTOR_LOG_PATH, COLLECTOR_PATH, COLLECTOR_STATE_PATH, COLLECTOR_STOP_PATH,
    DATABASE_PATH, RUNTIME_DIR, VENV_PYTHON,
)


@dataclass(frozen=True)
class CollectorStatus:
    state: str
    running: bool
    owned: bool
    pid: int | None = None
    run_id: str | None = None
    machine_id: str | None = None
    started_at_utc: str | None = None
    last_sample_at: str | None = None
    message: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class CollectorControlError(RuntimeError):
    pass


class CollectorController:
    def __init__(
        self,
        database_path: Path = DATABASE_PATH,
        state_path: Path = COLLECTOR_STATE_PATH,
        stop_path: Path = COLLECTOR_STOP_PATH,
        log_path: Path = COLLECTOR_LOG_PATH,
        collector_path: Path = COLLECTOR_PATH,
        interval_seconds: float = 2.0,
    ) -> None:
        self.database_path = database_path.resolve()
        self.state_path = state_path.resolve()
        self.stop_path = stop_path.resolve()
        self.log_path = log_path.resolve()
        self.collector_path = collector_path.resolve()
        self.interval_seconds = float(interval_seconds)
        self._process: subprocess.Popen[str] | None = None
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

    def _read_state(self) -> dict[str, object] | None:
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None

    def _write_state(self, state: dict[str, object]) -> None:
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
        temporary.replace(self.state_path)

    def _owned_process(self, state: dict[str, object] | None) -> psutil.Process | None:
        if not state or not state.get("pid"):
            return None
        try:
            process = psutil.Process(int(state["pid"]))
            if abs(process.create_time() - float(state.get("process_create_time", -1))) > 1.0:
                return None
            command = [str(value) for value in process.cmdline()]
            if str(self.collector_path) not in command or str(self.database_path) not in command:
                return None
            if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
                return None
            return process
        except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError, TypeError):
            return None

    def _find_exact_external_process(self) -> psutil.Process | None:
        try:
            processes = psutil.process_iter(["pid", "cmdline", "status"])
        except (PermissionError, psutil.Error):
            return None
        try:
            iterator = iter(processes)
        except (PermissionError, psutil.Error):
            return None
        while True:
            try:
                process = next(iterator)
                command = [str(value) for value in (process.info.get("cmdline") or [])]
                if str(self.collector_path) in command and str(self.database_path) in command:
                    if process.status() != psutil.STATUS_ZOMBIE:
                        return process
            except StopIteration:
                break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            except PermissionError:
                return None
        return None

    def _database_run(self, launched_at_utc: str | None = None) -> dict[str, object] | None:
        if not self.database_path.exists():
            return None
        try:
            with sqlite3.connect(self.database_path, timeout=3) as connection:
                connection.row_factory = sqlite3.Row
                query = """
                    SELECT r.run_id,r.machine_id,r.started_at_utc,r.status,
                           (SELECT MAX(timestamp) FROM system_metrics m WHERE m.run_id=r.run_id) AS last_sample_at
                    FROM runs r WHERE r.status='running'
                """
                params: tuple[object, ...] = ()
                if launched_at_utc:
                    query += " AND r.started_at_utc >= ?"
                    params = (launched_at_utc,)
                query += " ORDER BY r.started_at_utc DESC LIMIT 1"
                row = connection.execute(query, params).fetchone()
                return dict(row) if row else None
        except sqlite3.Error:
            return None

    def status(self) -> CollectorStatus:
        state = self._read_state()
        process = self._owned_process(state)
        if process:
            run = self._database_run(str(state.get("launched_at_utc") or "")) or {}
            started = str(run.get("started_at_utc") or state.get("launched_at_utc") or "")
            process_state = "Running" if run.get("run_id") else "Starting"
            return CollectorStatus(
                process_state, True, True, process.pid, run.get("run_id"), run.get("machine_id"),
                started or None, run.get("last_sample_at"),
                "Collector is recording live metrics." if run.get("run_id") else "Collector process is starting.",
            )
        external = self._find_exact_external_process()
        if external:
            run = self._database_run() or {}
            return CollectorStatus(
                "Running", True, False, external.pid, run.get("run_id"), run.get("machine_id"),
                run.get("started_at_utc"), run.get("last_sample_at"),
                "An exact collector process is already running outside this dashboard session.",
            )
        if state:
            return CollectorStatus("Error", False, True, message="The dashboard-owned collector exited unexpectedly. Check the collector log.")
        return CollectorStatus("Stopped", False, False, message="Collector is stopped.")

    def start(self) -> CollectorStatus:
        current = self.status()
        if current.running:
            return current
        if not self.collector_path.exists():
            raise CollectorControlError(f"Collector script is missing: {self.collector_path}")
        python = VENV_PYTHON if VENV_PYTHON.exists() else Path(sys.executable)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.stop_path.unlink(missing_ok=True)
        command = [
            str(python), str(self.collector_path), "monitor", "--db", str(self.database_path),
            "--interval", str(self.interval_seconds), "--quiet", "--stop-file", str(self.stop_path),
        ]
        launched_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        with self.log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n[{launched_at}] dashboard start: {' '.join(command)}\n")
            log.flush()
            process = subprocess.Popen(
                command, cwd=str(self.collector_path.parent.parent), stdout=log, stderr=log,
                stdin=subprocess.DEVNULL, start_new_session=True,
            )
        self._process = process
        state = {
            "pid": process.pid, "process_create_time": psutil.Process(process.pid).create_time(),
            "launched_at_utc": launched_at, "database_path": str(self.database_path),
            "collector_path": str(self.collector_path), "command": command,
        }
        self._write_state(state)
        for _ in range(30):
            result = self.status()
            if result.state in {"Running", "Error"}:
                return result
            time.sleep(0.1)
        return self.status()

    def stop(self, timeout_seconds: float = 12.0) -> CollectorStatus:
        state = self._read_state()
        process = self._owned_process(state)
        if not process:
            external = self._find_exact_external_process()
            if external:
                raise CollectorControlError("The collector was not started by this dashboard, so it will not be stopped here.")
            self.state_path.unlink(missing_ok=True)
            self.stop_path.unlink(missing_ok=True)
            return CollectorStatus("Stopped", False, False, message="Collector is already stopped.")
        self.stop_path.write_text("stop\n", encoding="utf-8")
        try:
            process.wait(timeout=timeout_seconds)
        except psutil.TimeoutExpired as exc:
            raise CollectorControlError("Collector did not stop cleanly within the safety timeout.") from exc
        finally:
            self.stop_path.unlink(missing_ok=True)
        if self._process is not None:
            self._process.wait(timeout=1)
            self._process = None
        self.state_path.unlink(missing_ok=True)
        return CollectorStatus("Stopped", False, False, message="Collection stopped cleanly; history was preserved.")
