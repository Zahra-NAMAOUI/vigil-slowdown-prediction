"""AdoptAI cross-platform system metric collector and experiment runner."""

from __future__ import annotations

import argparse
import ctypes
import csv
import hashlib
import json
import math
import multiprocessing
import os
import platform
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import uuid
from queue import Empty, Queue
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import psutil


VERSION = "2.3.0"
SCHEMA_VERSION = 4
DEFAULT_INTERVAL = 2.0
DEFAULT_LATENCY_HOST = "1.1.1.1"
DEFAULT_LATENCY_PORT = 443
METRICS_TABLE = "system_metrics"
V2_METRICS_TABLE = "samples"
PAWNIO_SETUP_URL = (
    "https://github.com/namazso/PawnIO.Setup/releases/download/2.2.0/"
    "PawnIO_setup.exe"
)
PAWNIO_SETUP_SHA256 = "1f519a22e47187f70a1379a48ca604981c4fcf694f4e65b734aaa74a9fba3032"

LEGACY_METRICS = [
    "machine_id", "timestamp", "cpu_pct", "ram_pct", "ram_used_mb",
    "swap_pct", "swap_used_mb", "disk_usage_pct", "disk_free_gb",
    "disk_read_mb_s", "disk_write_mb_s", "disk_latency_ms",
    "net_sent_mb_s", "net_recv_mb_s", "network_latency_ms",
    "process_count", "thread_count", "context_switches_per_s",
    "temperature_c", "gpu_usage_pct",
]

METRIC_COLUMNS = [
    "run_id", "machine_id", "timestamp", "elapsed_seconds", "phase",
    "stress_cpu_target_pct", "stress_memory_target_mb",
    "cpu_pct", "cpu_per_core_json", "cpu_frequency_mhz",
    "ram_pct", "ram_used_mb", "ram_available_mb", "swap_pct", "swap_used_mb",
    "disk_usage_pct", "disk_free_gb", "disk_read_mb_s", "disk_write_mb_s",
    "disk_latency_ms", "net_sent_mb_s", "net_recv_mb_s", "network_latency_ms",
    "process_count", "thread_count", "context_switches_per_s",
    "temperature_c", "gpu_usage_pct", "gpu_per_device_json",
    "battery_pct", "battery_plugged",
    "sensor_errors_json", "missed_deadline", "legacy_id",
]


def base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def bundled_resource(relative: str) -> Path:
    root = Path(getattr(sys, "_MEIPASS", base_dir()))
    return root / relative


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def parse_duration(value: str | float | int | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (float, int)):
        return float(value)
    text = value.strip().lower()
    multiplier = 1.0
    if text.endswith("ms"):
        multiplier, text = 0.001, text[:-2]
    elif text.endswith("s"):
        text = text[:-1]
    elif text.endswith("m"):
        multiplier, text = 60.0, text[:-1]
    elif text.endswith("h"):
        multiplier, text = 3600.0, text[:-1]
    try:
        result = float(text) * multiplier
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid duration: {value}") from exc
    if result < 0:
        raise argparse.ArgumentTypeError("duration cannot be negative")
    return result


def stable_machine_id() -> str:
    raw = f"{platform.node()}|{platform.system()}|{platform.machine()}"
    return hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()[:24]


def is_windows_admin() -> bool:
    if platform.system() != "Windows":
        return os.geteuid() == 0 if hasattr(os, "geteuid") else False
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


class Database:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, timeout=10)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.execute("PRAGMA journal_mode=WAL")

    def initialize(self) -> None:
        self._backup_before_schema_upgrade()
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS schema_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                machine_id TEXT NOT NULL,
                mode TEXT NOT NULL,
                started_at_utc TEXT NOT NULL,
                ended_at_utc TEXT,
                status TEXT NOT NULL,
                stop_reason TEXT,
                os_name TEXT NOT NULL,
                os_version TEXT NOT NULL,
                architecture TEXT NOT NULL,
                hostname TEXT NOT NULL,
                python_version TEXT NOT NULL,
                collector_version TEXT NOT NULL,
                sample_interval_seconds REAL NOT NULL,
                config_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL REFERENCES runs(run_id),
                timestamp_utc TEXT NOT NULL,
                event_type TEXT NOT NULL,
                phase TEXT,
                message TEXT NOT NULL,
                details_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE IF NOT EXISTS capabilities (
                run_id TEXT NOT NULL REFERENCES runs(run_id),
                metric_name TEXT NOT NULL,
                available INTEGER NOT NULL,
                provider TEXT,
                detail TEXT,
                PRIMARY KEY(run_id, metric_name)
            );
            """
        )
        self._migrate_measurements_to_v3()
        self.connection.execute(
            "INSERT OR REPLACE INTO schema_meta(key, value) VALUES('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        self.connection.commit()

    def _table_exists(self, name: str) -> bool:
        return self.connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone() is not None

    def _table_columns(self, name: str) -> set[str]:
        return {row[1] for row in self.connection.execute(f'PRAGMA table_info("{name}")')}

    def _backup_before_schema_upgrade(self) -> None:
        """Create one consistent SQLite backup before any measurement-schema upgrade."""
        has_measurements = self._table_exists(V2_METRICS_TABLE) or self._table_exists(METRICS_TABLE)
        if not has_measurements:
            return
        current_version = 0
        if self._table_exists("schema_meta"):
            row = self.connection.execute(
                "SELECT value FROM schema_meta WHERE key='schema_version'"
            ).fetchone()
            try:
                current_version = int(row[0]) if row else 0
            except (TypeError, ValueError):
                current_version = 0
        legacy_layout = self._table_exists(V2_METRICS_TABLE)
        if self._table_exists(METRICS_TABLE):
            legacy_layout = legacy_layout or "run_id" not in self._table_columns(METRICS_TABLE)
        if current_version >= SCHEMA_VERSION and not legacy_layout:
            return
        backup_path = self.path.with_name(
            f"{self.path.stem}.pre-v{SCHEMA_VERSION}.backup{self.path.suffix}"
        )
        if backup_path.exists():
            return
        self.connection.commit()
        backup = sqlite3.connect(backup_path)
        try:
            self.connection.backup(backup)
        finally:
            backup.close()

    def _create_canonical_metrics_table(self) -> None:
        self.connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {METRICS_TABLE} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL REFERENCES runs(run_id),
                machine_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                elapsed_seconds REAL,
                phase TEXT NOT NULL,
                stress_cpu_target_pct REAL,
                stress_memory_target_mb REAL,
                cpu_pct REAL,
                cpu_per_core_json TEXT,
                cpu_frequency_mhz REAL,
                ram_pct REAL,
                ram_used_mb REAL,
                ram_available_mb REAL,
                swap_pct REAL,
                swap_used_mb REAL,
                disk_usage_pct REAL,
                disk_free_gb REAL,
                disk_read_mb_s REAL,
                disk_write_mb_s REAL,
                disk_latency_ms REAL,
                net_sent_mb_s REAL,
                net_recv_mb_s REAL,
                network_latency_ms REAL,
                process_count INTEGER,
                thread_count INTEGER,
                context_switches_per_s REAL,
                temperature_c REAL,
                gpu_usage_pct REAL,
                gpu_per_device_json TEXT,
                battery_pct REAL,
                battery_plugged INTEGER,
                sensor_errors_json TEXT NOT NULL DEFAULT '{{}}',
                missed_deadline INTEGER NOT NULL DEFAULT 0,
                legacy_id INTEGER
            )
            """
        )
        self.connection.execute(
            f"CREATE UNIQUE INDEX IF NOT EXISTS idx_system_metrics_legacy "
            f"ON {METRICS_TABLE}(legacy_id) WHERE legacy_id IS NOT NULL"
        )
        self.connection.execute(
            f"CREATE INDEX IF NOT EXISTS idx_system_metrics_run_ts "
            f"ON {METRICS_TABLE}(run_id, timestamp)"
        )
        self.connection.execute(
            f"CREATE INDEX IF NOT EXISTS idx_system_metrics_phase ON {METRICS_TABLE}(phase)"
        )
        if "gpu_per_device_json" not in self._table_columns(METRICS_TABLE):
            self.connection.execute(
                f"ALTER TABLE {METRICS_TABLE} ADD COLUMN gpu_per_device_json TEXT"
            )

    def _import_legacy_table(self, table: str) -> int:
        columns = self._table_columns(table)
        if not set(LEGACY_METRICS + ["id"]).issubset(columns):
            raise RuntimeError(f"legacy metric table has an incompatible schema: {table}")
        rows = self.connection.execute(
            f'SELECT id, {", ".join(LEGACY_METRICS)} FROM "{table}"'
        ).fetchall()
        machines: dict[str, str] = {}
        imported = 0
        for row in rows:
            if self.connection.execute(
                f"SELECT 1 FROM {METRICS_TABLE} WHERE legacy_id=?", (row["id"],)
            ).fetchone():
                continue
            hostname = row["machine_id"] or "unknown"
            if hostname not in machines:
                run_id = "legacy-" + hashlib.sha256(hostname.encode()).hexdigest()[:16]
                machines[hostname] = run_id
                self.connection.execute(
                    """INSERT OR IGNORE INTO runs
                    (run_id,machine_id,mode,started_at_utc,status,os_name,os_version,architecture,
                     hostname,python_version,collector_version,sample_interval_seconds,config_json)
                    VALUES(?,?,?,'legacy','legacy','unknown','unknown','unknown',?,'unknown','1.x',10,'{}')""",
                    (run_id, stable_machine_id(), "legacy", hostname),
                )
            timestamp = str(row["timestamp"] or "")
            if timestamp and not timestamp.endswith("Z") and "+" not in timestamp:
                timestamp = timestamp.replace(" ", "T") + "Z"
            values = {
                "run_id": machines[hostname], "machine_id": hostname,
                "timestamp": timestamp, "elapsed_seconds": None, "phase": "legacy",
                "stress_cpu_target_pct": None, "stress_memory_target_mb": None,
                "cpu_pct": row["cpu_pct"], "cpu_per_core_json": None, "cpu_frequency_mhz": None,
                "ram_pct": row["ram_pct"], "ram_used_mb": row["ram_used_mb"],
                "ram_available_mb": None, "swap_pct": row["swap_pct"],
                "swap_used_mb": row["swap_used_mb"], "disk_usage_pct": row["disk_usage_pct"],
                "disk_free_gb": row["disk_free_gb"], "disk_read_mb_s": row["disk_read_mb_s"],
                "disk_write_mb_s": row["disk_write_mb_s"], "disk_latency_ms": row["disk_latency_ms"],
                "net_sent_mb_s": row["net_sent_mb_s"], "net_recv_mb_s": row["net_recv_mb_s"],
                "network_latency_ms": row["network_latency_ms"], "process_count": row["process_count"],
                "thread_count": row["thread_count"],
                "context_switches_per_s": row["context_switches_per_s"],
                "temperature_c": row["temperature_c"], "gpu_usage_pct": row["gpu_usage_pct"],
                "gpu_per_device_json": None,
                "battery_pct": None, "battery_plugged": None,
                "sensor_errors_json": json.dumps({"source": "legacy_import"}),
                "missed_deadline": 0, "legacy_id": row["id"],
            }
            self.insert_metric(values, commit=False)
            imported += 1
        return imported

    def _migrate_measurements_to_v3(self) -> None:
        """Unify legacy system_metrics and v2 samples into one canonical metric table."""
        legacy_temp = "_system_metrics_v1_migration"
        system_exists = self._table_exists(METRICS_TABLE)
        system_is_canonical = system_exists and "run_id" in self._table_columns(METRICS_TABLE)
        v2_exists = self._table_exists(V2_METRICS_TABLE)

        if system_is_canonical and not v2_exists:
            self._create_canonical_metrics_table()
            return

        with self.connection:
            if system_exists and not system_is_canonical:
                if self._table_exists(legacy_temp):
                    raise RuntimeError("unfinished v3 migration table already exists")
                self.connection.execute(f"ALTER TABLE {METRICS_TABLE} RENAME TO {legacy_temp}")

            self._create_canonical_metrics_table()

            copied_v2 = 0
            if v2_exists:
                source_count = self.connection.execute(
                    f"SELECT COUNT(*) FROM {V2_METRICS_TABLE}"
                ).fetchone()[0]
                target_columns = ",".join(METRIC_COLUMNS)
                source_table_columns = self._table_columns(V2_METRICS_TABLE)
                source_columns = ",".join(
                    "timestamp_utc" if column == "timestamp"
                    else column if column in source_table_columns
                    else "NULL"
                    for column in METRIC_COLUMNS
                )
                self.connection.execute(
                    f"INSERT INTO {METRICS_TABLE}(id,{target_columns}) "
                    f"SELECT id,{source_columns} FROM {V2_METRICS_TABLE} ORDER BY id"
                )
                copied_v2 = self.connection.execute(
                    f"SELECT COUNT(*) FROM {METRICS_TABLE}"
                ).fetchone()[0]
                if copied_v2 != source_count:
                    raise RuntimeError(
                        f"v3 migration count mismatch: expected {source_count}, copied {copied_v2}"
                    )

            imported_legacy = 0
            if self._table_exists(legacy_temp):
                imported_legacy = self._import_legacy_table(legacy_temp)

            final_count = self.connection.execute(
                f"SELECT COUNT(*) FROM {METRICS_TABLE}"
            ).fetchone()[0]
            if final_count != copied_v2 + imported_legacy:
                raise RuntimeError("v3 migration integrity check failed")

            if v2_exists:
                self.connection.execute(f"DROP TABLE {V2_METRICS_TABLE}")
            if self._table_exists(legacy_temp):
                self.connection.execute(f"DROP TABLE {legacy_temp}")
            self.connection.execute(
                "INSERT OR REPLACE INTO schema_meta(key,value) VALUES('v3_migrated_rows',?)",
                (str(final_count),),
            )

    def start_run(self, mode: str, interval: float, config: dict[str, Any]) -> str:
        run_id = str(uuid.uuid4())
        self.connection.execute(
            """INSERT INTO runs
            (run_id,machine_id,mode,started_at_utc,status,os_name,os_version,architecture,
             hostname,python_version,collector_version,sample_interval_seconds,config_json)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (run_id, stable_machine_id(), mode, utc_now(), "running", platform.system(),
             platform.version(), platform.machine(), platform.node(), platform.python_version(),
             VERSION, interval, json.dumps(config, sort_keys=True)),
        )
        self.connection.commit()
        return run_id

    def finish_run(self, run_id: str, status: str, reason: str) -> None:
        self.connection.execute(
            "UPDATE runs SET ended_at_utc=?,status=?,stop_reason=? WHERE run_id=?",
            (utc_now(), status, reason, run_id),
        )
        self.connection.commit()

    def insert_metric(self, values: dict[str, Any], commit: bool = True) -> None:
        placeholders = ",".join("?" for _ in METRIC_COLUMNS)
        self.connection.execute(
            f"INSERT OR IGNORE INTO {METRICS_TABLE}({','.join(METRIC_COLUMNS)}) VALUES({placeholders})",
            tuple(values.get(column) for column in METRIC_COLUMNS),
        )
        if commit:
            self.connection.commit()

    def event(self, run_id: str, event_type: str, message: str, phase: str | None = None,
              details: dict[str, Any] | None = None) -> None:
        self.connection.execute(
            "INSERT INTO events(run_id,timestamp_utc,event_type,phase,message,details_json) VALUES(?,?,?,?,?,?)",
            (run_id, utc_now(), event_type, phase, message, json.dumps(details or {}, sort_keys=True)),
        )
        self.connection.commit()

    def save_capabilities(self, run_id: str, capabilities: dict[str, tuple[bool, str, str]]) -> None:
        self.connection.executemany(
            "INSERT OR REPLACE INTO capabilities(run_id,metric_name,available,provider,detail) VALUES(?,?,?,?,?)",
            [(run_id, name, int(data[0]), data[1], data[2]) for name, data in capabilities.items()],
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()


class TemperatureProvider:
    """Persistent temperature reader with a Windows hardware-sensor helper."""

    def __init__(self):
        self.process: subprocess.Popen[str] | None = None
        self.lines: Queue[str] = Queue()
        self.provider = "psutil"
        self.sensor_detected = callable(getattr(psutil, "sensors_temperatures", None))
        self.fallback_value: float | None = None
        self.fallback_checked_at = 0.0
        self.last_diagnostic = "not_checked"
        self.gpu_detected = False
        self.gpu_provider = "none"
        self.gpu_devices: list[dict[str, Any]] = []
        self.gpu_usage: float | None = None
        self.macmon_process: subprocess.Popen[str] | None = None
        self.macmon_payload: dict[str, Any] = {}
        self.macmon_lock = threading.Lock()
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            self._start_macmon()
        helper = bundled_resource("sensor_helper/AdoptAITemperature.exe")
        if platform.system() == "Windows" and not helper.exists():
            helper = base_dir() / "vendor" / "librehardwaremonitor" / "AdoptAITemperature.exe"
        if platform.system() == "Windows" and helper.exists():
            self._start_helper(helper)
            if self.process:
                try:
                    self.sensor_detected = self.read() is not None
                    if not self.sensor_detected and not self._pawnio_installed():
                        if self._install_pawnio():
                            self.close()
                            self.lines = Queue()
                            self._start_helper(helper)
                            self.sensor_detected = bool(self.process) and self.read() is not None
                except Exception:
                    self.sensor_detected = False

    def _start_macmon(self) -> None:
        """Read real Apple Silicon temperature and GPU utilization without sudo."""
        executable = shutil.which("macmon")
        if not executable:
            self.last_diagnostic = "macmon_not_installed (install with: brew install macmon)"
            return
        try:
            self.macmon_process = subprocess.Popen(
                [executable, "pipe", "--interval", "1000"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1,
            )
            assert self.macmon_process.stdout is not None

            def read_macmon() -> None:
                assert self.macmon_process and self.macmon_process.stdout
                for line in self.macmon_process.stdout:
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(payload, dict):
                        continue
                    with self.macmon_lock:
                        self.macmon_payload = payload

            threading.Thread(
                target=read_macmon, name="macmon-reader", daemon=True
            ).start()
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                with self.macmon_lock:
                    if self.macmon_payload:
                        break
                if self.macmon_process.poll() is not None:
                    raise RuntimeError("macmon exited before producing a sample")
                time.sleep(0.05)
            with self.macmon_lock:
                payload = dict(self.macmon_payload)
            temperature = (payload.get("temp") or {}).get("cpu_temp_avg")
            if temperature is None:
                raise RuntimeError("macmon did not return CPU temperature")
            self.sensor_detected = True
            self.provider = "macmon"
            self.last_diagnostic = "ok"
        except Exception as exc:
            self.last_diagnostic = f"macmon_failed: {exc}"
            self._close_macmon()

    @staticmethod
    def _pawnio_installed() -> bool:
        if platform.system() != "Windows":
            return False
        try:
            import winreg

            path = r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\PawnIO"
            views = (winreg.KEY_WOW64_64KEY, winreg.KEY_WOW64_32KEY)
            for view in views:
                try:
                    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path, 0,
                                        winreg.KEY_READ | view):
                        return True
                except FileNotFoundError:
                    continue
        except (ImportError, OSError):
            pass
        return False

    def _install_pawnio(self) -> bool:
        """Install the signed low-level sensor backend from its official release."""
        print("Installing signed Windows temperature support (PawnIO 2.2.0)...")
        target = Path(tempfile.gettempdir()) / "AdoptAI" / "PawnIO_setup_2.2.0.exe"
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            if (not target.exists() or
                    hashlib.sha256(target.read_bytes()).hexdigest().lower() != PAWNIO_SETUP_SHA256):
                with urllib.request.urlopen(PAWNIO_SETUP_URL, timeout=45) as response:
                    payload = response.read()
                digest = hashlib.sha256(payload).hexdigest().lower()
                if digest != PAWNIO_SETUP_SHA256:
                    raise RuntimeError("PawnIO installer checksum did not match the official release")
                target.write_bytes(payload)

            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            subprocess.Popen(
                [str(target), "-install", "-silent"],
                creationflags=flags,
            )
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                if self._pawnio_installed():
                    time.sleep(1)
                    print("Windows temperature support installed successfully.")
                    return True
                time.sleep(1)
            raise RuntimeError("temperature driver installation did not complete")
        except Exception as exc:
            self.last_diagnostic = f"pawnio_install_failed: {exc}"
            print(f"Temperature support installation failed: {exc}", file=sys.stderr)
            return False

    def _start_helper(self, helper: Path) -> None:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            self.process = subprocess.Popen(
                [str(helper)], cwd=str(helper.parent), stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                bufsize=1, creationflags=flags,
            )
            assert self.process.stdout is not None

            def read_lines() -> None:
                assert self.process and self.process.stdout
                for line in self.process.stdout:
                    self.lines.put(line.strip())

            threading.Thread(target=read_lines, name="temperature-reader", daemon=True).start()
            if self.lines.get(timeout=15) != "READY":
                raise RuntimeError("temperature helper did not become ready")
            self.provider = "LibreHardwareMonitor"
        except Exception:
            self.close()

    def read(self) -> float | None:
        if self.macmon_process and self.macmon_process.poll() is None:
            with self.macmon_lock:
                payload = dict(self.macmon_payload)
            value = (payload.get("temp") or {}).get("cpu_temp_avg")
            if value is not None:
                self.last_diagnostic = "ok"
                return float(value)
            self.last_diagnostic = "macmon_returned_no_cpu_temperature"
            return None
        if self.process and self.process.poll() is None and self.process.stdin:
            self.process.stdin.write("READ\n")
            self.process.stdin.flush()
            try:
                payload = json.loads(self.lines.get(timeout=5))
            except (Empty, json.JSONDecodeError) as exc:
                raise RuntimeError(f"temperature helper response failed: {exc}") from exc
            devices = payload.get("gpu_devices")
            if isinstance(devices, list):
                normalized: list[dict[str, Any]] = []
                for device in devices:
                    if not isinstance(device, dict) or device.get("usage_pct") is None:
                        continue
                    normalized.append({
                        "name": str(device.get("name") or "GPU"),
                        "usage_pct": round(max(0.0, min(100.0, float(device["usage_pct"]))), 1),
                    })
                self.gpu_devices = normalized
                self.gpu_usage = max(
                    (device["usage_pct"] for device in normalized), default=None
                )
                self.gpu_detected = bool(normalized)
                if self.gpu_detected:
                    self.gpu_provider = str(payload.get("gpu_source") or "LibreHardwareMonitor")
            value = payload.get("temperature_c")
            if value is not None:
                self.sensor_detected = True
                self.provider = payload.get("source") or "LibreHardwareMonitor"
                self.last_diagnostic = "ok"
                return float(value)
            self.last_diagnostic = payload.get("diagnostic") or "helper_returned_no_sensor"
            fallback = self._windows_fallback()
            if fallback is not None:
                self.sensor_detected = True
                return fallback
            return None
        function = getattr(psutil, "sensors_temperatures", None)
        if not callable(function):
            return None
        groups = function() or {}
        preferred: list[float] = []
        fallback: list[float] = []
        for group, sensors in groups.items():
            for sensor in sensors:
                current = getattr(sensor, "current", None)
                if current is None:
                    continue
                value = float(current)
                fallback.append(value)
                label = f"{group} {getattr(sensor, 'label', '')}".lower()
                if any(token in label for token in ("cpu", "core", "package", "k10temp", "zenpower")):
                    preferred.append(value)
        values = preferred or fallback
        return max(values) if values else None

    def read_gpu(self) -> tuple[float | None, list[dict[str, Any]], str]:
        if self.macmon_process and self.macmon_process.poll() is None:
            with self.macmon_lock:
                payload = dict(self.macmon_payload)
            ratio = payload.get("gpu_active_ratio")
            if ratio is None:
                gpu = payload.get("gpu_usage")
                if isinstance(gpu, list) and len(gpu) > 1:
                    ratio = gpu[1]
            if ratio is not None:
                usage = round(max(0.0, min(100.0, float(ratio) * 100.0)), 1)
                devices = [{"name": "Apple Silicon GPU", "usage_pct": usage}]
                self.gpu_usage = usage
                self.gpu_devices = devices
                self.gpu_detected = True
                self.gpu_provider = "macmon"
                return usage, devices, self.gpu_provider
        return self.gpu_usage, list(self.gpu_devices), self.gpu_provider

    def _windows_fallback(self) -> float | None:
        """Try standard and third-party WMI thermal providers already present on Windows."""
        if platform.system() != "Windows":
            return None
        now = time.monotonic()
        if now - self.fallback_checked_at < 10:
            return self.fallback_value
        self.fallback_checked_at = now
        script = r"""
$values = @()
foreach ($ns in @('root/LibreHardwareMonitor','root/OpenHardwareMonitor')) {
  try {
    Get-CimInstance -Namespace $ns -ClassName Sensor -ErrorAction Stop |
      Where-Object { $_.SensorType -eq 'Temperature' -and
        (($_.Name + ' ' + $_.Identifier) -match 'CPU|Core|Package|Processor|DTS|Tctl|Tdie') } |
      ForEach-Object { if ($null -ne $_.Value) { $values += [double]$_.Value } }
  } catch {}
}
try {
  Get-CimInstance -Namespace root/wmi -ClassName MSAcpi_ThermalZoneTemperature -ErrorAction Stop |
    ForEach-Object {
      $c = ([double]$_.CurrentTemperature / 10.0) - 273.15
      if ($c -gt 0 -and $c -lt 130) { $values += $c }
    }
} catch {}
try {
  Get-CimInstance -Namespace root/cimv2 -ClassName Win32_PerfFormattedData_Counters_ThermalZoneInformation -ErrorAction Stop |
    ForEach-Object {
      $c = ([double]$_.Temperature / 10.0) - 273.15
      if ($c -gt 0 -and $c -lt 130) { $values += $c }
    }
} catch {}
if ($values.Count -gt 0) { ($values | Measure-Object -Maximum).Maximum }
"""
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True, text=True, timeout=8, creationflags=flags, check=False,
        )
        try:
            self.fallback_value = float(result.stdout.strip().splitlines()[-1])
            self.provider = "Windows WMI thermal provider"
            self.last_diagnostic = "ok"
        except (ValueError, IndexError):
            self.fallback_value = None
            self.last_diagnostic = "no_supported_cpu_sensor_from_any_windows_provider"
        return self.fallback_value

    @property
    def available(self) -> bool:
        return self.sensor_detected

    def close(self) -> None:
        self._close_macmon()
        process, self.process = self.process, None
        if not process:
            return
        try:
            if process.poll() is None and process.stdin:
                process.stdin.write("QUIT\n")
                process.stdin.flush()
                process.wait(timeout=3)
        except Exception:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=2)
        finally:
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream:
                    stream.close()

    def _close_macmon(self) -> None:
        process, self.macmon_process = self.macmon_process, None
        if not process:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        for stream in (process.stdout, process.stderr):
            if stream:
                stream.close()


class MetricCollector:
    def __init__(self, latency_host: str = DEFAULT_LATENCY_HOST,
                 latency_port: int = DEFAULT_LATENCY_PORT,
                 latency_timeout: float = 1.0):
        self.latency_host = latency_host
        self.latency_port = latency_port
        self.latency_timeout = latency_timeout
        self.previous_time: float | None = None
        self.previous_disk: Any = None
        self.previous_net: Any = None
        self.previous_context: int | None = None
        self.temperature_provider = TemperatureProvider()
        self.gpu_provider = self.temperature_provider.gpu_provider
        self.gpu_available = self.temperature_provider.gpu_detected
        psutil.cpu_percent(interval=None)

    def capabilities(self) -> dict[str, tuple[bool, str, str]]:
        temperature = self.temperature_provider.available
        if not self.gpu_available:
            usage, devices, provider = self._gpu_metrics()
            self.gpu_available = usage is not None or bool(devices)
            self.gpu_provider = provider
        battery = callable(getattr(psutil, "sensors_battery", None))
        return {
            "cpu": (True, "psutil", "portable"),
            "memory": (True, "psutil", "portable"),
            "disk": (True, "psutil", "portable counters when exposed by the OS"),
            "network": (True, "psutil/socket", "TCP latency does not require ICMP privileges"),
            "temperature": (temperature, self.temperature_provider.provider,
                            "CPU sensor; scheduled stress requires an actual reading"),
            "gpu": (self.gpu_available, self.gpu_provider,
                    "Intel/AMD/NVIDIA; gpu_usage_pct is the busiest device and "
                    "gpu_per_device_json contains every detected GPU"),
            "battery": (battery, "psutil", "empty on desktop systems"),
        }

    @staticmethod
    def _attempt(name: str, function: Callable[[], Any], errors: dict[str, str], default: Any = None) -> Any:
        try:
            return function()
        except Exception as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"
            return default

    def _temperature(self) -> float | None:
        return self.temperature_provider.read()

    def close(self) -> None:
        self.temperature_provider.close()

    @staticmethod
    def _nvidia_gpu_usage() -> tuple[float | None, list[dict[str, Any]], str]:
        executable = shutil.which("nvidia-smi")
        if not executable:
            return None, [], "none"
        result = subprocess.run(
            [executable, "--query-gpu=name,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2, check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "nvidia-smi failed")
        devices: list[dict[str, Any]] = []
        for line in result.stdout.splitlines():
            if not line.strip() or "," not in line:
                continue
            name, value = line.rsplit(",", 1)
            devices.append({
                "name": name.strip(),
                "usage_pct": round(max(0.0, min(100.0, float(value.strip()))), 1),
            })
        usage = max((device["usage_pct"] for device in devices), default=None)
        return usage, devices, "nvidia-smi" if devices else "none"

    @staticmethod
    def _windows_gpu_usage() -> tuple[float | None, list[dict[str, Any]], str]:
        if platform.system() != "Windows":
            return None, [], "none"
        script = r"""
$engineTotals = @{}
try {
  $rows = Get-CimInstance -Namespace root/cimv2 `
    -ClassName Win32_PerfFormattedData_GPUPerformanceCounters_GPUEngine `
    -ErrorAction Stop
  foreach ($row in $rows) {
    $instance = [string]$row.Name
    if ($instance -eq '_Total') { continue }
    if ($instance -match 'luid_(0x[0-9A-Fa-f]+_0x[0-9A-Fa-f]+)_phys_([0-9]+)_eng_([0-9]+)_engtype_(.+)$') {
      $device = $Matches[1] + '_phys_' + $Matches[2]
      $engine = $device + '_eng_' + $Matches[3]
      $value = [double]$row.UtilizationPercentage
      if (-not $engineTotals.ContainsKey($engine)) { $engineTotals[$engine] = 0.0 }
      $engineTotals[$engine] += $value
    }
  }
} catch {}
$deviceTotals = @{}
foreach ($entry in $engineTotals.GetEnumerator()) {
  if ($entry.Key -match '^(.*)_eng_[0-9]+$') {
    $device = $Matches[1]
    $value = [Math]::Min(100.0, [double]$entry.Value)
    if (-not $deviceTotals.ContainsKey($device) -or $value -gt $deviceTotals[$device]) {
      $deviceTotals[$device] = $value
    }
  }
}
$controllerNames = @()
try {
  $controllerNames = @(Get-CimInstance Win32_VideoController -ErrorAction Stop |
    ForEach-Object { [string]$_.Name })
} catch {}
$devices = @()
$keys = @($deviceTotals.Keys | Sort-Object)
for ($index = 0; $index -lt $keys.Count; $index++) {
  $name = if ($index -lt $controllerNames.Count) { $controllerNames[$index] } else { $keys[$index] }
  $devices += [PSCustomObject]@{
    name = $name
    usage_pct = [Math]::Round([double]$deviceTotals[$keys[$index]], 1)
  }
}
$maximum = $null
if ($devices.Count -gt 0) {
  $maximum = ($devices | Measure-Object -Property usage_pct -Maximum).Maximum
}
[PSCustomObject]@{ gpu_usage_pct = $maximum; gpu_devices = $devices } |
  ConvertTo-Json -Compress -Depth 4
"""
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True, text=True, timeout=8, creationflags=flags, check=False,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None, [], "none"
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        raw_devices = payload.get("gpu_devices") or []
        if isinstance(raw_devices, dict):
            raw_devices = [raw_devices]
        devices = [
            {
                "name": str(item.get("name") or "GPU"),
                "usage_pct": round(max(0.0, min(100.0, float(item["usage_pct"]))), 1),
            }
            for item in raw_devices
            if isinstance(item, dict) and item.get("usage_pct") is not None
        ]
        usage = max((device["usage_pct"] for device in devices), default=None)
        return usage, devices, "Windows GPU Engine" if devices else "none"

    def _gpu_metrics(self) -> tuple[float | None, list[dict[str, Any]], str]:
        usage, devices, provider = self.temperature_provider.read_gpu()
        if devices:
            self.gpu_available, self.gpu_provider = True, provider
            return usage, devices, provider
        usage, devices, provider = self._windows_gpu_usage()
        if not devices:
            usage, devices, provider = self._nvidia_gpu_usage()
        self.gpu_available = usage is not None or bool(devices)
        self.gpu_provider = provider
        return usage, devices, provider

    def _network_latency(self) -> float | None:
        start = time.perf_counter()
        with socket.create_connection((self.latency_host, self.latency_port), self.latency_timeout):
            return round((time.perf_counter() - start) * 1000, 3)

    @staticmethod
    def _thread_count() -> int:
        total = 0
        for process in psutil.process_iter(["num_threads"]):
            try:
                total += process.info.get("num_threads") or 0
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return total

    def collect(self) -> dict[str, Any]:
        errors: dict[str, str] = {}
        now = time.monotonic()
        elapsed = now - self.previous_time if self.previous_time is not None else None
        memory = self._attempt("memory", psutil.virtual_memory, errors)
        swap = self._attempt("swap", psutil.swap_memory, errors)
        disk_path = os.environ.get("SystemDrive", "C:") + "\\" if platform.system() == "Windows" else "/"
        disk_usage = self._attempt("disk_usage", lambda: psutil.disk_usage(disk_path), errors)
        disk = self._attempt("disk_io", psutil.disk_io_counters, errors)
        net = self._attempt("network_io", psutil.net_io_counters, errors)
        context = self._attempt("context_switches", lambda: psutil.cpu_stats().ctx_switches, errors)
        frequency = self._attempt("cpu_frequency", psutil.cpu_freq, errors)
        battery_function = getattr(psutil, "sensors_battery", None)
        battery = self._attempt("battery", battery_function, errors) if callable(battery_function) else None

        def rate(current: Any, previous: Any, attribute: str, divisor: float = 1.0) -> float | None:
            if elapsed is None or elapsed <= 0 or current is None or previous is None:
                return None
            value = getattr(current, attribute, None)
            old = getattr(previous, attribute, None)
            if value is None or old is None or value < old:
                return None
            return round((value - old) / elapsed / divisor, 4)

        read_rate = rate(disk, self.previous_disk, "read_bytes", 1_000_000)
        write_rate = rate(disk, self.previous_disk, "write_bytes", 1_000_000)
        sent_rate = rate(net, self.previous_net, "bytes_sent", 1_000_000)
        recv_rate = rate(net, self.previous_net, "bytes_recv", 1_000_000)
        context_rate = None
        if elapsed and context is not None and self.previous_context is not None and context >= self.previous_context:
            context_rate = round((context - self.previous_context) / elapsed, 1)
        disk_latency = None
        if elapsed and disk is not None and self.previous_disk is not None:
            read_ops = getattr(disk, "read_count", 0) - getattr(self.previous_disk, "read_count", 0)
            write_ops = getattr(disk, "write_count", 0) - getattr(self.previous_disk, "write_count", 0)
            time_delta = ((getattr(disk, "read_time", 0) - getattr(self.previous_disk, "read_time", 0)) +
                          (getattr(disk, "write_time", 0) - getattr(self.previous_disk, "write_time", 0)))
            operations = read_ops + write_ops
            if operations > 0 and time_delta >= 0:
                disk_latency = round(time_delta / operations, 3)

        temperature = self._attempt("temperature", self._temperature, errors)
        if temperature is None and "temperature" not in errors:
            errors["temperature"] = self.temperature_provider.last_diagnostic
        gpu_usage, gpu_devices, _gpu_provider = self._attempt(
            "gpu", self._gpu_metrics, errors, (None, [], "none")
        )
        result = {
            "cpu_pct": self._attempt("cpu", lambda: psutil.cpu_percent(interval=None), errors),
            "cpu_per_core_json": json.dumps(self._attempt("cpu_per_core", lambda: psutil.cpu_percent(interval=None, percpu=True), errors)),
            "cpu_frequency_mhz": getattr(frequency, "current", None) if frequency else None,
            "ram_pct": getattr(memory, "percent", None),
            "ram_used_mb": round(memory.used / 1_000_000, 1) if memory else None,
            "ram_available_mb": round(memory.available / 1_000_000, 1) if memory else None,
            "swap_pct": getattr(swap, "percent", None),
            "swap_used_mb": round(swap.used / 1_000_000, 1) if swap else None,
            "disk_usage_pct": getattr(disk_usage, "percent", None),
            "disk_free_gb": round(disk_usage.free / 1024 ** 3, 2) if disk_usage else None,
            "disk_read_mb_s": read_rate, "disk_write_mb_s": write_rate,
            "disk_latency_ms": disk_latency, "net_sent_mb_s": sent_rate,
            "net_recv_mb_s": recv_rate,
            "network_latency_ms": self._attempt("network_latency", self._network_latency, errors),
            "process_count": self._attempt("process_count", lambda: len(psutil.pids()), errors),
            "thread_count": self._attempt("thread_count", self._thread_count, errors),
            "context_switches_per_s": context_rate,
            "temperature_c": temperature,
            "gpu_usage_pct": gpu_usage,
            "gpu_per_device_json": json.dumps(gpu_devices, sort_keys=True),
            "battery_pct": getattr(battery, "percent", None) if battery else None,
            "battery_plugged": int(battery.power_plugged) if battery and battery.power_plugged is not None else None,
            "sensor_errors_json": json.dumps(errors, sort_keys=True),
        }
        self.previous_time, self.previous_disk, self.previous_net = now, disk, net
        self.previous_context = context
        return result


def _cpu_worker(load_pct: float, stop_event: Any) -> None:
    window = 0.1
    busy = window * max(0.0, min(load_pct, 100.0)) / 100.0
    while not stop_event.is_set():
        start = time.perf_counter()
        while time.perf_counter() - start < busy and not stop_event.is_set():
            _ = 137 * 139
        remaining = window - (time.perf_counter() - start)
        if remaining > 0:
            stop_event.wait(remaining)


def _memory_worker(byte_count: int, stop_event: Any, ready_queue: Any) -> None:
    try:
        block = bytearray(byte_count)
        for index in range(0, len(block), 4096):
            block[index] = 1
        ready_queue.put((True, len(block)))
        stop_event.wait()
    except BaseException as exc:
        ready_queue.put((False, f"{type(exc).__name__}: {exc}"))


class StressController:
    def __init__(self):
        self.context = multiprocessing.get_context("spawn")
        self.stop_event: Any = None
        self.processes: list[multiprocessing.Process] = []
        self.memory_mb = 0.0

    def start(self, cpu_pct: float, memory_mb: float) -> None:
        self.stop()
        self.stop_event = self.context.Event()
        logical = psutil.cpu_count(logical=True) or 1
        worker_count = max(1, min(logical, math.ceil(logical * cpu_pct / 100.0)))
        per_worker_load = min(100.0, cpu_pct * logical / worker_count)
        for _ in range(worker_count):
            process = self.context.Process(target=_cpu_worker, args=(per_worker_load, self.stop_event), daemon=True)
            process.start()
            self.processes.append(process)
        if memory_mb > 0:
            ready = self.context.Queue()
            process = self.context.Process(
                target=_memory_worker,
                args=(int(memory_mb * 1024 * 1024), self.stop_event, ready), daemon=True,
            )
            process.start()
            self.processes.append(process)
            try:
                ok, detail = ready.get(timeout=15)
            except Exception as exc:
                self.stop()
                raise RuntimeError(f"memory worker did not initialize: {exc}") from exc
            if not ok:
                self.stop()
                raise RuntimeError(f"memory allocation failed: {detail}")
            self.memory_mb = detail / 1024 / 1024

    def failed_process(self) -> multiprocessing.Process | None:
        for process in self.processes:
            if not process.is_alive() and process.exitcode is not None:
                return process
        return None

    def stop(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()
        deadline = time.monotonic() + 5.0
        for process in self.processes:
            process.join(timeout=max(0.0, deadline - time.monotonic()))
        for process in self.processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
        self.processes.clear()
        self.stop_event = None
        self.memory_mb = 0.0


@dataclass
class SafetyLimits:
    temperature_c: float = 90.0
    memory_pct: float = 90.0
    minimum_available_mb: float = 1024.0
    maximum_stress_memory_mb: float = 4096.0


class Runner:
    def __init__(self, database: Database, interval: float, latency_host: str,
                 latency_port: int, quiet: bool = False):
        self.database = database
        self.interval = interval
        self.collector = MetricCollector(latency_host, latency_port)
        self.quiet = quiet
        self.stop_requested = False
        self.run_id = ""
        self.run_started = 0.0
        self.stress: StressController | None = None

    def request_stop(self, *_: Any) -> None:
        self.stop_requested = True

    def install_signal_handlers(self) -> None:
        signal.signal(signal.SIGINT, self.request_stop)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, self.request_stop)

    def begin(self, mode: str, config: dict[str, Any]) -> None:
        self.run_id = self.database.start_run(mode, self.interval, config)
        self.database.save_capabilities(self.run_id, self.collector.capabilities())
        self.run_started = time.monotonic()
        self.install_signal_handlers()

    def collect_sample(self, phase: str, cpu_target: float | None = None,
                       memory_target: float | None = None, missed: bool = False) -> dict[str, Any]:
        values = self.collector.collect()
        values.update({
            "run_id": self.run_id, "machine_id": stable_machine_id(), "timestamp": utc_now(),
            "elapsed_seconds": round(time.monotonic() - self.run_started, 3), "phase": phase,
            "stress_cpu_target_pct": cpu_target, "stress_memory_target_mb": memory_target,
            "missed_deadline": int(missed), "legacy_id": None,
        })
        self.database.insert_metric(values)
        if not self.quiet:
            temperature = values["temperature_c"]
            temperature_text = f"{temperature}C" if temperature is not None else "unavailable"
            gpu = values["gpu_usage_pct"]
            gpu_text = f"{gpu}%" if gpu is not None else "unavailable"
            print(
                f"[{values['timestamp']}] phase={phase} CPU={values['cpu_pct']}% "
                f"RAM={values['ram_pct']}% temp={temperature_text} GPU={gpu_text} "
                f"net={values['network_latency_ms']}ms"
            )
        return values

    def run_phase(self, phase: str, duration: float | None, cpu_target: float | None = None,
                  memory_target: float | None = None,
                  safety: SafetyLimits | None = None) -> str | None:
        self.database.event(self.run_id, "phase_started", f"Started {phase}", phase)
        started = time.monotonic()
        deadline = started
        while not self.stop_requested and (duration is None or time.monotonic() - started < duration):
            now = time.monotonic()
            missed = now > deadline + self.interval * 0.5
            values = self.collect_sample(phase, cpu_target, memory_target, missed)
            if safety and phase.startswith("stress"):
                reason = safety_reason(values, safety)
                if reason:
                    self.database.event(self.run_id, "safety_stop", reason, phase)
                    return reason
                if self.stress:
                    failed = self.stress.failed_process()
                    if failed:
                        reason = f"stress worker failed with exit code {failed.exitcode}"
                        self.database.event(self.run_id, "worker_failure", reason, phase)
                        return reason
            deadline += self.interval
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
        self.database.event(self.run_id, "phase_finished", f"Finished {phase}", phase)
        return "interrupted" if self.stop_requested else None

    def finish(self, status: str, reason: str) -> None:
        if self.stress:
            self.stress.stop()
        self.collector.close()
        if self.run_id:
            self.database.finish_run(self.run_id, status, reason)


def safety_reason(values: dict[str, Any], limits: SafetyLimits) -> str | None:
    temperature = values.get("temperature_c")
    if temperature is not None and temperature >= limits.temperature_c:
        return f"temperature reached {temperature:.1f}C (limit {limits.temperature_c:.1f}C)"
    memory = values.get("ram_pct")
    if memory is not None and memory >= limits.memory_pct:
        return f"RAM usage reached {memory:.1f}% (limit {limits.memory_pct:.1f}%)"
    available = values.get("ram_available_mb")
    if available is not None and available < limits.minimum_available_mb:
        return f"available RAM fell to {available:.1f} MB (minimum {limits.minimum_available_mb:.1f} MB)"
    return None


def run_monitor(args: argparse.Namespace) -> int:
    database = Database(args.db)
    database.initialize()
    runner = Runner(database, args.interval, args.latency_host, args.latency_port, args.quiet)
    runner.begin("monitor", vars_for_json(args))
    print(f"AdoptAI monitor {VERSION} started; run={runner.run_id}; database={args.db}")
    try:
        reason = runner.run_phase("monitor", args.duration)
        status = "interrupted" if reason == "interrupted" else "completed"
        runner.finish(status, reason or "duration completed")
        return 130 if reason == "interrupted" else 0
    except BaseException as exc:
        runner.finish("failed", f"{type(exc).__name__}: {exc}")
        raise
    finally:
        database.close()


def check_power(allow_battery: bool) -> None:
    function = getattr(psutil, "sensors_battery", None)
    battery = function() if callable(function) else None
    if battery and battery.power_plugged is False and not allow_battery:
        raise RuntimeError("experiment refused while running on battery; connect AC power or use --allow-battery")


def run_experiment(args: argparse.Namespace) -> int:
    check_power(args.allow_battery)
    database = Database(args.db)
    database.initialize()
    runner = Runner(database, args.interval, args.latency_host, args.latency_port, args.quiet)
    runner.stress = StressController()
    limits = SafetyLimits(args.max_temperature, args.max_memory_pct, args.min_available_mb, args.max_stress_memory_mb)
    runner.begin("experiment", vars_for_json(args))
    print(f"AdoptAI experiment {VERSION} started; run={runner.run_id}; database={args.db}")
    reason: str | None = None
    try:
        reason = runner.run_phase("baseline", args.baseline_seconds)
        if not reason:
            preflight = runner.collect_sample("baseline")
            if preflight.get("temperature_c") is None:
                reason = "CPU temperature unavailable; stress was not started"
                runner.database.event(runner.run_id, "stress_blocked", reason, "baseline")
        stages = [(40.0, 0.0), (65.0, 0.10), (80.0, 0.20)]
        for index, (cpu, memory_fraction) in enumerate(stages, start=1):
            if reason:
                break
            available_mb = psutil.virtual_memory().available / 1024 / 1024
            memory_mb = min(available_mb * memory_fraction, limits.maximum_stress_memory_mb)
            phase = "stress_cpu" if memory_mb <= 0 else "stress_cpu_ram"
            runner.database.event(runner.run_id, "stress_stage", f"Stage {index}: CPU {cpu}%, RAM {memory_mb:.1f} MB", phase)
            runner.stress.start(cpu, memory_mb)
            reason = runner.run_phase(phase, args.stage_seconds, cpu, round(memory_mb, 1), limits)
            runner.stress.stop()
        if not runner.stop_requested:
            recovery_reason = runner.run_phase("recovery", args.recovery_seconds)
            reason = reason or recovery_reason
        status = "completed" if reason is None else ("interrupted" if reason == "interrupted" else "safety_stopped")
        runner.finish(status, reason or "experiment completed")
        return 0 if status == "completed" else (130 if status == "interrupted" else 2)
    except BaseException as exc:
        runner.finish("failed", f"{type(exc).__name__}: {exc}")
        raise
    finally:
        runner.stress.stop()
        database.close()


def stress_preflight_reason(values: dict[str, Any], start_temperature: float,
                            allow_battery: bool, start_memory_pct: float = 88.0) -> str | None:
    temperature = values.get("temperature_c")
    if temperature is None:
        return "CPU temperature unavailable"
    if temperature >= start_temperature:
        return f"CPU temperature is {temperature:.1f}C; waiting below {start_temperature:.1f}C"
    if values.get("ram_pct") is not None and values["ram_pct"] >= start_memory_pct:
        return f"RAM usage is already {values['ram_pct']:.1f}%"
    if values.get("battery_plugged") == 0 and not allow_battery:
        return "computer is running on battery"
    return None


def run_day(args: argparse.Namespace) -> int:
    """Collect all day and periodically create safe labelled stress windows."""
    database = Database(args.db)
    database.initialize()
    runner = Runner(database, args.interval, args.latency_host, args.latency_port, args.quiet)
    runner.stress = StressController()
    limits = SafetyLimits(args.max_temperature, args.max_memory_pct,
                          args.min_available_mb, args.max_stress_memory_mb)
    runner.begin("day", vars_for_json(args))
    started = time.monotonic()

    def remaining() -> float:
        return max(0.0, args.duration - (time.monotonic() - started))

    def bounded(requested: float) -> float:
        return min(requested, remaining())

    print(
        f"AdoptAI all-day collection {VERSION} started; run={runner.run_id}; "
        f"duration={args.duration / 3600:.1f}h; database={args.db}"
    )
    interrupted = False
    cycles = 0
    try:
        reason = runner.run_phase("baseline", bounded(args.initial_baseline_seconds))
        interrupted = reason == "interrupted"
        while remaining() > 0 and not runner.stop_requested:
            preflight = runner.collect_sample("monitor")
            blocked = stress_preflight_reason(preflight, args.stress_start_temperature,
                                              args.allow_battery, args.stress_start_memory_pct)
            if blocked:
                runner.database.event(runner.run_id, "stress_postponed", blocked, "monitor")
                print(f"Stress postponed: {blocked}. Normal collection continues.")
                reason = runner.run_phase("monitor", bounded(args.retry_seconds))
                interrupted = reason == "interrupted"
                if interrupted:
                    break
                continue

            cycles += 1
            runner.database.event(runner.run_id, "stress_cycle_started",
                                  f"Scheduled stress cycle {cycles}", "stress_cpu")
            cycle_stop: str | None = None
            for index, (cpu, memory_fraction) in enumerate(
                    [(40.0, 0.0), (65.0, 0.10), (80.0, 0.20)], start=1):
                if remaining() <= 0 or runner.stop_requested:
                    break
                available_mb = psutil.virtual_memory().available / 1024 / 1024
                memory_mb = min(available_mb * memory_fraction,
                                limits.maximum_stress_memory_mb)
                phase = "stress_cpu" if memory_mb <= 0 else "stress_cpu_ram"
                runner.database.event(
                    runner.run_id, "stress_stage",
                    f"Cycle {cycles}, stage {index}: CPU {cpu}%, RAM {memory_mb:.1f} MB", phase,
                )
                runner.stress.start(cpu, memory_mb)
                cycle_stop = runner.run_phase(
                    phase, bounded(args.stage_seconds), cpu, round(memory_mb, 1), limits,
                )
                runner.stress.stop()
                if cycle_stop:
                    break

            recovery = runner.run_phase("recovery", bounded(args.recovery_seconds))
            if cycle_stop and cycle_stop != "interrupted":
                runner.database.event(runner.run_id, "cycle_safety_stop", cycle_stop, "recovery")
            interrupted = cycle_stop == "interrupted" or recovery == "interrupted"
            if interrupted or remaining() <= 0:
                break
            runner.database.event(
                runner.run_id, "cooldown_started",
                f"Normal collection for {args.cooldown_seconds / 60:.1f} minutes before next stress",
                "monitor",
            )
            cooldown = runner.run_phase("monitor", bounded(args.cooldown_seconds))
            interrupted = cooldown == "interrupted"
            if interrupted:
                break

        status = "interrupted" if interrupted else "completed"
        message = "stopped by user" if interrupted else f"all-day run completed with {cycles} stress cycles"
        runner.finish(status, message)
        return 130 if interrupted else 0
    except BaseException as exc:
        runner.finish("failed", f"{type(exc).__name__}: {exc}")
        raise
    finally:
        runner.stress.stop()
        database.close()


def run_doctor(args: argparse.Namespace) -> int:
    collector = MetricCollector(args.latency_host, args.latency_port)
    print(f"AdoptAI {VERSION}")
    print(f"OS: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Python: {platform.python_version()}; psutil: {psutil.__version__}")
    if platform.system() == "Windows":
        print(f"Administrator: {'yes' if is_windows_admin() else 'no'}")
    print(f"Machine ID: {stable_machine_id()}")
    print("Capabilities:")
    for name, (available, provider, detail) in collector.capabilities().items():
        print(f"  {'OK' if available else '--':2} {name:12} provider={provider:10} {detail}")
    sample = collector.collect()
    errors = json.loads(sample["sensor_errors_json"])
    print("One-sample check:")
    for name in ("cpu_pct", "ram_pct", "disk_usage_pct", "network_latency_ms", "temperature_c", "gpu_usage_pct"):
        print(f"  {name:24} {sample.get(name)}")
    if errors:
        print("Non-fatal collection errors:")
        for name, error in errors.items():
            print(f"  {name}: {error}")
    collector.close()
    return 0


def run_export(args: argparse.Namespace) -> int:
    if not args.db.exists():
        raise FileNotFoundError(f"database does not exist: {args.db}")
    database = Database(args.db)
    database.initialize()
    query = f"SELECT * FROM {METRICS_TABLE}"
    parameters: tuple[Any, ...] = ()
    if args.run_id:
        query += " WHERE run_id=?"
        parameters = (args.run_id,)
    query += " ORDER BY timestamp,id"
    rows = database.connection.execute(query, parameters)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([description[0] for description in rows.description])
        for row in rows:
            writer.writerow(tuple(row))
            count += 1
    database.close()
    print(f"Exported {count} system-metric rows to {args.output}")
    return 0


def vars_for_json(args: argparse.Namespace) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items() if key != "handler"}


def normalize_frozen_multiprocessing_argv() -> None:
    """Remove an executable-path argument injected by some frozen spawn builds."""
    if not getattr(sys, "frozen", False) or len(sys.argv) < 2:
        return
    try:
        first = Path(sys.argv[1]).resolve()
        executable = Path(sys.executable).resolve()
    except (OSError, ValueError):
        return
    if first == executable:
        del sys.argv[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AdoptAI cross-platform metric collector")
    parser.add_argument("--version", action="version", version=f"AdoptAI {VERSION}")
    subparsers = parser.add_subparsers(dest="command")

    def common(command: argparse.ArgumentParser, include_db: bool = True) -> None:
        if include_db:
            command.add_argument("--db", type=Path, default=base_dir() / "data" / "metrics.db")
        command.add_argument("--latency-host", default=DEFAULT_LATENCY_HOST)
        command.add_argument("--latency-port", type=int, default=DEFAULT_LATENCY_PORT)

    monitor = subparsers.add_parser("monitor", help="collect continuously or for a fixed duration")
    common(monitor)
    monitor.add_argument("--interval", type=float, default=DEFAULT_INTERVAL)
    monitor.add_argument("--duration", type=parse_duration)
    monitor.add_argument("--quiet", action="store_true")
    monitor.set_defaults(handler=run_monitor)

    experiment = subparsers.add_parser("experiment", help="run a labelled baseline/stress/recovery experiment")
    common(experiment)
    experiment.add_argument("--interval", type=float, default=DEFAULT_INTERVAL)
    experiment.add_argument("--baseline-seconds", type=parse_duration, default=300.0)
    experiment.add_argument("--stage-seconds", type=parse_duration, default=120.0)
    experiment.add_argument("--recovery-seconds", type=parse_duration, default=300.0)
    experiment.add_argument("--max-temperature", type=float, default=90.0)
    experiment.add_argument("--max-memory-pct", type=float, default=90.0)
    experiment.add_argument("--min-available-mb", type=float, default=1024.0)
    experiment.add_argument("--max-stress-memory-mb", type=float, default=4096.0)
    experiment.add_argument("--allow-battery", action="store_true")
    experiment.add_argument("--quiet", action="store_true")
    experiment.set_defaults(handler=run_experiment)

    day = subparsers.add_parser(
        "day", help="collect for a workday with periodic safe stress and recovery windows"
    )
    common(day)
    day.add_argument("--interval", type=float, default=DEFAULT_INTERVAL)
    day.add_argument("--duration", type=parse_duration, default=8 * 3600.0)
    day.add_argument("--initial-baseline-seconds", type=parse_duration, default=30 * 60.0)
    day.add_argument("--stage-seconds", type=parse_duration, default=120.0)
    day.add_argument("--recovery-seconds", type=parse_duration, default=10 * 60.0)
    day.add_argument("--cooldown-seconds", type=parse_duration, default=90 * 60.0)
    day.add_argument("--retry-seconds", type=parse_duration, default=5 * 60.0)
    day.add_argument("--stress-start-temperature", type=float, default=75.0)
    day.add_argument("--stress-start-memory-pct", type=float, default=88.0)
    day.add_argument("--max-temperature", type=float, default=90.0)
    day.add_argument("--max-memory-pct", type=float, default=90.0)
    day.add_argument("--min-available-mb", type=float, default=1024.0)
    day.add_argument("--max-stress-memory-mb", type=float, default=4096.0)
    day.add_argument("--allow-battery", action="store_true")
    day.add_argument("--quiet", action="store_true")
    day.set_defaults(handler=run_day)

    doctor = subparsers.add_parser("doctor", help="show platform and metric availability")
    common(doctor, include_db=False)
    doctor.set_defaults(handler=run_doctor)

    export = subparsers.add_parser("export", help="export system metrics to CSV")
    export.add_argument("--db", type=Path, default=base_dir() / "data" / "metrics.db")
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--run-id")
    export.set_defaults(handler=run_export)
    return parser


def main(argv: list[str] | None = None) -> int:
    normalize_frozen_multiprocessing_argv()
    multiprocessing.freeze_support()
    arguments = list(sys.argv[1:] if argv is None else argv)
    automatic = not arguments
    if automatic:
        arguments.insert(0, "day")
    parser = build_parser()
    args = parser.parse_args(arguments)
    if getattr(args, "interval", DEFAULT_INTERVAL) <= 0:
        parser.error("--interval must be greater than zero")
    try:
        result = int(args.handler(args))
        if automatic:
            output = base_dir() / "data" / "dataset.csv"
            run_export(argparse.Namespace(db=args.db, output=output, run_id=None))
            print("\nAdoptAI is finished and your dataset is ready:")
            print(f"  Database: {args.db}")
            print(f"  CSV file: {output}")
            if getattr(sys, "frozen", False) and sys.stdin and sys.stdin.isatty():
                try:
                    input("Press Enter to close...")
                except EOFError:
                    pass
        return result
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
