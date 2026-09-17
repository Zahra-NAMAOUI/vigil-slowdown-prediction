"""Integration-focused tests for the V1 dashboard without changing ML artifacts."""

from __future__ import annotations

import json
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from dashboard.collector_control import CollectorController
from dashboard.config import COLLECTOR_PATH, PROJECT_ROOT
from dashboard.diagnostic import training_parity_diagnostic
from dashboard.history import (
    initialize_dashboard_tables,
    list_runs,
    load_alerts,
    load_metrics_history,
    load_prediction_history,
    log_prediction,
    session_sample_count,
)
from dashboard.inference import InferenceResult, load_inference_engine
from dashboard.live_features import RAW_CANDIDATE_FEATURES, assign_live_segments, prepare_current_feature_row
from dashboard.ui_components import risk_card, theme_css


def synthetic_history(seconds: int = 120, interval: int = 2) -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01T00:00:00Z", periods=seconds // interval + 1, freq=f"{interval}s")
    rows = len(timestamps)
    frame = pd.DataFrame({
        "id": np.arange(1, rows + 1), "run_id": "live-run", "machine_id": "machine-live",
        "timestamp": timestamps.astype(str), "elapsed_seconds": np.arange(rows) * interval, "phase": "monitor",
    })
    defaults = {
        "cpu_pct": np.linspace(20, 60, rows), "cpu_frequency_mhz": 2400.0,
        "ram_pct": np.linspace(45, 55, rows), "ram_used_mb": 8000.0, "ram_available_mb": 8000.0,
        "swap_pct": 4.0, "swap_used_mb": 500.0, "disk_usage_pct": 58.0, "disk_free_gb": 200.0,
        "disk_read_mb_s": 1.0, "disk_write_mb_s": 0.5, "disk_latency_ms": 1.2,
        "net_sent_mb_s": 0.1, "net_recv_mb_s": 0.2, "network_latency_ms": 20.0,
        "process_count": 250, "thread_count": 3000, "context_switches_per_s": 12000.0,
        "battery_pct": np.nan, "battery_plugged": np.nan,
    }
    for column in RAW_CANDIDATE_FEATURES:
        frame[column] = defaults[column]
    return frame


class FeatureAndInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = load_inference_engine()

    def test_warmup_and_gap_reset(self) -> None:
        short = synthetic_history(118)
        self.assertFalse(prepare_current_feature_row(short, self.engine.input_feature_names).ready)
        complete = synthetic_history(120)
        ready = prepare_current_feature_row(complete, self.engine.input_feature_names)
        self.assertTrue(ready.ready)
        self.assertEqual(ready.feature_row.shape, (1, 228))
        after_gap = complete.copy()
        extra = complete.iloc[[-1]].copy()
        extra["id"] = int(complete.id.max()) + 1
        extra["timestamp"] = str(pd.Timestamp(complete.timestamp.iloc[-1]) + pd.Timedelta(seconds=20))
        after_gap = pd.concat([after_gap, extra], ignore_index=True)
        self.assertFalse(prepare_current_feature_row(after_gap, self.engine.input_feature_names).ready)

    def test_preprocessing_and_frozen_inference(self) -> None:
        ready = prepare_current_feature_row(synthetic_history(120), self.engine.input_feature_names)
        result = self.engine.predict(ready.feature_row)
        self.assertGreaterEqual(result.model_score, 0.0)
        self.assertLessEqual(result.model_score, 1.0)
        self.assertEqual(result.predicted_class, int(result.model_score >= 0.50))
        self.assertAlmostEqual(result.risk_score, result.model_score * 100)

    def test_exact_training_feature_parity(self) -> None:
        result = training_parity_diagnostic(PROJECT_ROOT / "data/modeling/train.csv")
        self.assertTrue(result["feature_parity"]["exact_match"])
        self.assertEqual(result["feature_parity"]["mismatching_features"], 0)


class PersistenceTests(unittest.TestCase):
    def test_new_database_is_a_valid_empty_dashboard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "metrics.db"
            initialize_dashboard_tables(database)
            self.assertTrue(list_runs(database).empty)
            self.assertTrue(load_metrics_history(database, "not-yet-created").empty)
            self.assertEqual(session_sample_count(database), (None, 0))

    def test_session_sample_count_uses_requested_or_latest_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "metrics.db"
            with sqlite3.connect(database) as connection:
                connection.executescript(
                    """
                    CREATE TABLE runs (run_id TEXT PRIMARY KEY, started_at_utc TEXT);
                    CREATE TABLE system_metrics (id INTEGER PRIMARY KEY, run_id TEXT);
                    INSERT INTO runs VALUES ('older', '2026-01-01T00:00:00Z');
                    INSERT INTO runs VALUES ('latest', '2026-01-02T00:00:00Z');
                    INSERT INTO system_metrics(run_id) VALUES ('older'), ('latest'), ('latest');
                    """
                )
            self.assertEqual(session_sample_count(database, "older"), ("older", 1))
            self.assertEqual(session_sample_count(database), ("latest", 2))

    def test_prediction_and_alert_deduplication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "metrics.db"
            initialize_dashboard_tables(database)
            result = InferenceResult(0.73, 73.0, 1, "High Risk")
            inserted, alerted = log_prediction(database, "2026-01-01T00:00:00Z", "m", "r", result, 70.0, 80.0)
            self.assertTrue(inserted and alerted)
            inserted, alerted = log_prediction(database, "2026-01-01T00:00:00Z", "m", "r", result, 70.0, 80.0)
            self.assertFalse(inserted or alerted)
            inserted, alerted = log_prediction(database, "2026-01-01T00:00:02Z", "m", "r", result, 70.0, 80.0)
            self.assertTrue(inserted); self.assertFalse(alerted)
            self.assertEqual(len(load_prediction_history(database, "r")), 2)
            self.assertEqual(len(load_alerts(database, "r")), 1)


class PresentationTests(unittest.TestCase):
    def test_light_and_dark_theme_tokens_are_distinct(self) -> None:
        dark = theme_css("dark")
        light = theme_css("light")
        self.assertIn("#07111F", dark)
        self.assertIn("#F7F9FC", light)
        self.assertNotEqual(dark, light)

    def test_risk_scale_marker_uses_score_without_changing_prediction(self) -> None:
        class MarkdownCapture:
            body = ""

            def markdown(self, body: str, **_: object) -> None:
                self.body = body

        capture = MarkdownCapture()
        result = InferenceResult(0.73, 73.0, 1, "High Risk")
        risk_card(capture, result)
        self.assertIn('left:73.00%', capture.body)
        self.assertIn("High Risk", capture.body)
        self.assertEqual(result.threshold, 0.50)


class CollectorLifecycleTests(unittest.TestCase):
    def test_start_duplicate_stop_preserves_rows_and_new_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            controller = CollectorController(
                database_path=root / "metrics.db", state_path=root / "state.json",
                stop_path=root / "stop.flag", log_path=root / "collector.log",
                collector_path=COLLECTOR_PATH, interval_seconds=0.5,
            )
            first = controller.start()
            self.assertTrue(first.running)
            duplicate = controller.start()
            self.assertEqual(first.pid, duplicate.pid)
            deadline = time.time() + 15
            while time.time() < deadline:
                current = controller.status()
                if current.run_id and current.last_sample_at:
                    first = current; break
                time.sleep(0.25)
            self.assertIsNotNone(first.run_id)
            controller.stop()
            with sqlite3.connect(root / "metrics.db") as connection:
                first_rows = connection.execute("SELECT COUNT(*) FROM system_metrics WHERE run_id=?", (first.run_id,)).fetchone()[0]
                first_status = connection.execute("SELECT status FROM runs WHERE run_id=?", (first.run_id,)).fetchone()[0]
            self.assertGreater(first_rows, 0); self.assertEqual(first_status, "interrupted")
            second = controller.start()
            deadline = time.time() + 15
            while time.time() < deadline:
                current = controller.status()
                if current.run_id and current.run_id != first.run_id:
                    second = current; break
                time.sleep(0.25)
            self.assertNotEqual(first.run_id, second.run_id)
            controller.stop()
            with sqlite3.connect(root / "metrics.db") as connection:
                preserved = connection.execute("SELECT COUNT(*) FROM system_metrics WHERE run_id=?", (first.run_id,)).fetchone()[0]
            self.assertEqual(preserved, first_rows)


if __name__ == "__main__":
    unittest.main(verbosity=2)
