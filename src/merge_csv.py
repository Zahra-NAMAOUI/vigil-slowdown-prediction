#!/usr/bin/env python3
"""
Export the system_metrics table from the merged SQLite database
to a CSV file.

This script does not merge several CSV files.

Input:
    data/merged/metrics_db_fusionnees.db

Output:
    data/merged/metrics_db_fusionnees.csv
"""

from pathlib import Path
import sqlite3

import pandas as pd


# Project root:
# CO_ADOPTAI/
BASE_DIR = Path(__file__).resolve().parent.parent

# Folder containing the merged database.
MERGED_DIR = BASE_DIR / "data" / "merged"

# Input merged SQLite database.
INPUT_DB = MERGED_DIR / "merged_metrics.db"

# Output CSV file.
OUTPUT_CSV = MERGED_DIR / "merged_metrics.csv"

def main() -> None:
    """
    Read system_metrics from the merged database
    and export it to CSV.
    """

    # Create the merged folder if it does not exist.
    MERGED_DIR.mkdir(parents=True, exist_ok=True)

    # Verify that the merged DB exists.
    if not INPUT_DB.exists():
        raise SystemExit(
            f"Merged database not found:\n{INPUT_DB}\n\n"
            "Run merge_db.py first."
        )

    try:
        # Open the merged database in read-only mode.
        database_uri = INPUT_DB.resolve().as_uri() + "?mode=ro"

        with sqlite3.connect(
            database_uri,
            uri=True,
        ) as connection:

            # Verify that system_metrics exists.
            table_check = pd.read_sql_query(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table'
                  AND name = 'system_metrics'
                """,
                connection,
            )

            if table_check.empty:
                raise SystemExit(
                    "The system_metrics table does not exist "
                    "in the merged database."
                )

            # Read the complete system_metrics table.
            dataframe = pd.read_sql_query(
                """
                SELECT *
                FROM system_metrics
                """,
                connection,
            )

        if dataframe.empty:
            raise SystemExit(
                "The system_metrics table is empty."
            )

        # Sort the exported dataset.
        sort_columns = [
            column
            for column in [
                "machine_id",
                "run_id",
                "timestamp",
            ]
            if column in dataframe.columns
        ]

        if sort_columns:
            dataframe.sort_values(
                by=sort_columns,
                inplace=True,
            )

        dataframe.reset_index(
            drop=True,
            inplace=True,
        )

        # Export to CSV.
        dataframe.to_csv(
            OUTPUT_CSV,
            index=False,
        )

        print("CSV export completed successfully.")
        print(f"Rows exported: {len(dataframe)}")
        print(f"CSV file: {OUTPUT_CSV}")

    except sqlite3.Error as error:
        raise SystemExit(
            f"SQLite error: {error}"
        )

    except Exception as error:
        raise SystemExit(
            f"CSV export error: {error}"
        )


if __name__ == "__main__":
    main()