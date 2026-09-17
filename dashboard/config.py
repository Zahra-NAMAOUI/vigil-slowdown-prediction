"""Central paths and immutable V1 application constants."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _path_from_env(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default.resolve()


DATABASE_PATH = _path_from_env("ADOPTAI_DB_PATH", PROJECT_ROOT / "src/data/metrics.db")
MODEL_PATH = _path_from_env("ADOPTAI_MODEL_PATH", PROJECT_ROOT / "models/baseline/lightgbm.joblib")
PREPROCESSOR_PATH = _path_from_env(
    "ADOPTAI_PREPROCESSOR_PATH", PROJECT_ROOT / "models/preprocessing/tree_preprocessor.joblib"
)
MODEL_METADATA_PATH = PROJECT_ROOT / "models/final_v1/model_metadata.json"
FINAL_TEST_METRICS_PATH = PROJECT_ROOT / "reports/final_test_metrics.csv"
FEATURE_METADATA_PATH = PROJECT_ROOT / "reports/preprocessing_feature_report.csv"
COLLECTOR_PATH = PROJECT_ROOT / "src/collect_agent.py"
VENV_PYTHON = PROJECT_ROOT / "adoptai_env/bin/python"
RUNTIME_DIR = _path_from_env("ADOPTAI_RUNTIME_DIR", PROJECT_ROOT / "dashboard/.runtime")
COLLECTOR_STATE_PATH = RUNTIME_DIR / "collector_state.json"
COLLECTOR_STOP_PATH = RUNTIME_DIR / "collector.stop"
COLLECTOR_LOG_PATH = RUNTIME_DIR / "collector.log"

REFERENCE_THRESHOLD = 0.50
WARMUP_SECONDS = 120.0
ALERT_COOLDOWN_SECONDS = 300
REFRESH_SECONDS = 2
MAX_RECENT_ROWS = 1_000
HISTORY_MINUTES = 10

