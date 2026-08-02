#!/usr/bin/env python3
"""
clean.py
--------

Clean the `system_metrics` table from the merged SQLite database
and export the cleaned dataset as a CSV file for the AdoptAI
analysis and machine-learning stages.

Input:
    data/merged/merged_metrics.db

Output:
    data/processed/cleaned_metrics.csv

Important:
- The SQLite database is opened in read-only mode.
- The original database is never modified.
- All transformations are performed in memory.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd


# =========================================================
# PROJECT PATHS
# =========================================================

# Location of the current file:
# CO_ADOPTAI/src/clean.py
#
# .parent        -> CO_ADOPTAI/src/
# .parent.parent -> CO_ADOPTAI/
BASE_DIR = Path(__file__).resolve().parent.parent

# Merged database created by merge_db.py.
INPUT_DB = (
    BASE_DIR
    / "data"
    / "merged"
    / "merged_metrics.db"
)

# Table that contains the collected system measurements.
TABLE_NAME = "system_metrics"

# Final cleaned CSV dataset.
OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "cleaned_metrics.csv"
)


# =========================================================
# REQUIRED COLUMNS
# =========================================================

# These columns must exist in the system_metrics table.
REQUIRED_METRICS_COLUMNS = {
    "run_id",
    "machine_id",
    "timestamp",
    "missed_deadline",
    "sensor_errors_json",
}

# These columns must exist in the runs table.
REQUIRED_RUN_COLUMNS = {
    "run_id",
    "machine_id",
    "status",
    "ended_at_utc",
}


def validate_table_name(table_name: str) -> None:
    """
    Verify that the SQLite table name is valid.

    This prevents accidental or unsafe SQL table names.
    """

    if not table_name.replace("_", "").isalnum():
        raise ValueError(
            f"Invalid table name: {table_name!r}"
        )


def validate_columns(
    dataframe: pd.DataFrame,
    required_columns: set[str],
    dataframe_name: str,
) -> None:
    """
    Verify that all required columns exist in a DataFrame.
    """

    missing_columns = (
        required_columns - set(dataframe.columns)
    )

    if missing_columns:
        missing_text = ", ".join(
            sorted(missing_columns)
        )

        raise ValueError(
            f"Missing columns in {dataframe_name}: "
            f"{missing_text}"
        )


def load_tables(
    db_path: Path,
    table_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the system_metrics and runs tables.

    The SQLite database is opened in read-only mode, so this
    script cannot modify the original merged database.
    """

    validate_table_name(table_name)

    # Convert the local database path into a SQLite URI.
    db_uri = db_path.resolve().as_uri() + "?mode=ro"

    try:
        with sqlite3.connect(
            db_uri,
            uri=True,
        ) as connection:

            # Read all system measurements.
            metrics = pd.read_sql_query(
                f'SELECT * FROM "{table_name}"',
                connection,
            )

            # Read only the columns needed from the runs table.
            #
            # machine_id is renamed to run_machine_id to avoid
            # confusion with system_metrics.machine_id.
            runs = pd.read_sql_query(
                """
                SELECT
                    run_id,
                    machine_id AS run_machine_id,
                    status,
                    ended_at_utc
                FROM runs
                """,
                connection,
            )

    except sqlite3.Error as error:
        raise RuntimeError(
            "Error while reading the SQLite database: "
            f"{error}"
        ) from error

    # Verify the system_metrics columns.
    validate_columns(
        metrics,
        REQUIRED_METRICS_COLUMNS,
        table_name,
    )

    # After renaming machine_id, these are the expected
    # columns in the loaded runs DataFrame.
    required_loaded_run_columns = {
        "run_id",
        "run_machine_id",
        "status",
        "ended_at_utc",
    }

    validate_columns(
        runs,
        required_loaded_run_columns,
        "runs",
    )

    return metrics, runs


def fix_machine_id(
    dataframe: pd.DataFrame,
    runs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Correct machine_id using the runs table as the reference.

    For every run_id, the machine_id stored in the runs table
    is considered the trusted value.
    """

    result = dataframe.merge(
        runs[["run_id", "run_machine_id"]],
        on="run_id",
        how="left",
        validate="many_to_one",
    )

    # True when a machine reference exists in the runs table.
    valid_reference = result[
        "run_machine_id"
    ].notna()

    # Detect rows where system_metrics.machine_id differs from
    # the machine_id stored in the runs table.
    mismatch_mask = (
        valid_reference
        & result["machine_id"]
        .fillna("")
        .astype(str)
        .ne(
            result["run_machine_id"]
            .fillna("")
            .astype(str)
        )
    )

    mismatch_count = int(mismatch_mask.sum())

    missing_reference_count = int(
        (~valid_reference).sum()
    )

    if mismatch_count:
        print(
            f"[machine_id] {mismatch_count} rows corrected "
            "using the runs table."
        )

    if missing_reference_count:
        print(
            "[machine_id] Warning: "
            f"{missing_reference_count} rows do not have "
            "a matching run in the runs table."
        )

    # Use run_machine_id when available.
    # Otherwise, keep the original machine_id.
    result["machine_id"] = (
        result["run_machine_id"].combine_first(
            result["machine_id"]
        )
    )

    # Remove the temporary column.
    return result.drop(
        columns=["run_machine_id"]
    )


def normalize_timestamps(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert timestamp values into a consistent UTC datetime.
    """

    result = dataframe.copy()

    result["timestamp"] = pd.to_datetime(
        result["timestamp"],
        utc=True,
        format="mixed",
        errors="coerce",
    )

    # Invalid timestamps become NaT.
    invalid_count = int(
        result["timestamp"].isna().sum()
    )

    if invalid_count:
        print(
            f"[timestamp] Warning: {invalid_count} invalid "
            "timestamps were converted to NaT."
        )

    return result


def contains_sensor_error(
    raw_json: object,
) -> bool:
    """
    Return True when sensor_errors_json contains an error.

    Empty values such as {}, null or an empty string are
    considered to contain no sensor error.

    Invalid JSON is considered an error because its content
    cannot be trusted.
    """

    # Missing values are considered to contain no error.
    if raw_json is None:
        return False

    # Check pandas missing values safely.
    try:
        if pd.isna(raw_json):
            return False
    except (TypeError, ValueError):
        pass

    # A non-empty Python dictionary contains an error.
    if isinstance(raw_json, dict):
        return bool(raw_json)

    # A non-empty Python list also contains information
    # about an error.
    if isinstance(raw_json, list):
        return bool(raw_json)

    text = str(raw_json).strip()

    # These values represent no sensor error.
    if text in {
        "",
        "{}",
        "[]",
        "null",
        "None",
    }:
        return False

    try:
        parsed_json = json.loads(text)

    except (TypeError, json.JSONDecodeError):
        # Invalid JSON is considered unreliable.
        return True

    if isinstance(parsed_json, dict):
        return bool(parsed_json)

    if isinstance(parsed_json, list):
        return bool(parsed_json)

    return parsed_json not in (
        None,
        False,
        0,
        "",
    )


def add_reliability_flags(
    dataframe: pd.DataFrame,
    runs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add reliability indicators without deleting observations.

    Two new columns are created:

    sample_reliable:
        1 when the measurement is considered reliable.
        0 when it contains a collection problem.

    run_complete:
        1 when the corresponding collection run finished
        successfully.
        0 otherwise.
    """

    result = dataframe.copy()

    # Convert missed_deadline to numeric.
    #
    # Any non-zero value means that the collector did not
    # collect the sample at the expected time.
    missed_deadline = (
        pd.to_numeric(
            result["missed_deadline"],
            errors="coerce",
        )
        .fillna(0)
        .ne(0)
    )

    # Detect sensor errors stored as JSON.
    sensor_error = (
        result["sensor_errors_json"]
        .apply(contains_sensor_error)
    )

    # A missing or invalid timestamp is unreliable.
    invalid_timestamp = (
        result["timestamp"].isna()
    )

    # A missing run_id prevents linking the measurement
    # to its collection session.
    missing_run_id = (
        result["run_id"].isna()
    )

    # A sample is reliable only when none of the problems
    # above are present.
    result["sample_reliable"] = (
        ~(
            missed_deadline
            | sensor_error
            | invalid_timestamp
            | missing_run_id
        )
    ).astype("int8")

    unreliable_count = int(
        result["sample_reliable"].eq(0).sum()
    )

    total_count = len(result)

    unreliable_ratio = (
        unreliable_count / total_count
        if total_count
        else 0
    )

    print(
        f"[reliability] {unreliable_count}/{total_count} "
        "rows marked sample_reliable=0 "
        f"({unreliable_ratio:.1%})."
    )

    # Keep one run-status row for each run_id.
    run_status = (
        runs[
            [
                "run_id",
                "status",
                "ended_at_utc",
            ]
        ]
        .drop_duplicates(
            subset=["run_id"]
        )
    )

    # Add the run information to every measurement.
    result = result.merge(
        run_status,
        on="run_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_run"),
    )

    # Normalize status text for easier comparison.
    normalized_status = (
        result["status"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Convert run ending time to datetime.
    ended_at = pd.to_datetime(
        result["ended_at_utc"],
        utc=True,
        errors="coerce",
    )

    # Different possible words that indicate a successful run.
    successful_statuses = {
        "completed",
        "complete",
        "finished",
        "success",
        "succeeded",
    }

    # A run is complete when:
    # 1. Its status indicates success.
    # 2. It has a valid ending timestamp.
    result["run_complete"] = (
        normalized_status.isin(
            successful_statuses
        )
        & ended_at.notna()
    ).astype("int8")

    return result


def drop_duplicate_samples(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Remove duplicate measurements.

    Two rows are considered duplicates when they have the same:
    - run_id
    - timestamp

    When duplicates exist, the script keeps:
    1. The reliable row first.
    2. The row with fewer missing values.
    """

    if dataframe.empty:
        return dataframe

    result = dataframe.copy()

    # Count missing values in every row.
    result["_missing_value_count"] = (
        result.isna().sum(axis=1)
    )

    # Sort before removing duplicates.
    #
    # sample_reliable=False/0 goes after True/1.
    # Rows with fewer missing values come first.
    result = result.sort_values(
        by=[
            "run_id",
            "timestamp",
            "sample_reliable",
            "_missing_value_count",
        ],
        ascending=[
            True,
            True,
            False,
            True,
        ],
        na_position="last",
    )

    rows_before = len(result)

    result = result.drop_duplicates(
        subset=[
            "run_id",
            "timestamp",
        ],
        keep="first",
    )

    removed_count = (
        rows_before - len(result)
    )

    if removed_count:
        print(
            f"[duplicates] {removed_count} duplicate "
            "samples removed."
        )

    # Remove the temporary helper column.
    return result.drop(
        columns=["_missing_value_count"]
    )


def export_csv(
    dataframe: pd.DataFrame,
    output_file: Path,
) -> None:
    """
    Export the cleaned DataFrame to CSV.
    """

    # Create data/processed automatically if it is missing.
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        output_file,
        index=False,
        encoding="utf-8",
        na_rep="",
        date_format="%Y-%m-%dT%H:%M:%S.%fZ",
    )


def main() -> None:
    """
    Execute the complete cleaning pipeline.
    """

    print("Starting data cleaning...")
    print(f"Input database: {INPUT_DB}")
    print(f"Output dataset: {OUTPUT_FILE}\n")

    # Stop when the merged database does not exist.
    if not INPUT_DB.exists():
        raise SystemExit(
            "Merged database not found:\n"
            f"{INPUT_DB}\n\n"
            "Run this command first:\n"
            "python3 src/merge_db.py"
        )

    # Load the two required database tables.
    metrics, runs = load_tables(
        INPUT_DB,
        TABLE_NAME,
    )

    print(
        f"Rows loaded from {TABLE_NAME}: "
        f"{len(metrics)}"
    )

    print(
        f"Runs loaded: {len(runs)}"
    )

    # Export an empty CSV when system_metrics contains no rows.
    if metrics.empty:
        print(
            "[warning] The system_metrics table is empty."
        )

        export_csv(
            metrics,
            OUTPUT_FILE,
        )

        return

    # Step 1: correct machine identifiers.
    cleaned_data = fix_machine_id(
        metrics,
        runs,
    )

    # Step 2: convert timestamps to UTC datetime.
    cleaned_data = normalize_timestamps(
        cleaned_data
    )

    # Step 3: create sample and run reliability flags.
    cleaned_data = add_reliability_flags(
        cleaned_data,
        runs,
    )

    # Step 4: remove duplicate measurements.
    cleaned_data = drop_duplicate_samples(
        cleaned_data
    )

    # Step 5: organize rows by machine, run and time.
    cleaned_data = (
        cleaned_data
        .sort_values(
            by=[
                "machine_id",
                "run_id",
                "timestamp",
            ],
            na_position="last",
        )
        .reset_index(drop=True)
    )

    # Step 6: export the final cleaned CSV.
    export_csv(
        cleaned_data,
        OUTPUT_FILE,
    )

    print("\nCleaning completed successfully.")

    print(
        f"Output: {OUTPUT_FILE}"
    )

    print(
        f"Final shape: "
        f"{len(cleaned_data)} rows × "
        f"{len(cleaned_data.columns)} columns"
    )


if __name__ == "__main__":
    main()