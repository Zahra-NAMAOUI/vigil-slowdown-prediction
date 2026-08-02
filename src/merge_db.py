#!/usr/bin/env python3
"""
Merge all SQLite databases stored in data/raw/db/.

Outputs:
1. data/merged/metrics_db_fusionnees.db
   Contains all merged database tables.

2. data/merged/metrics_db_fusionnees.csv
   Contains only the merged system_metrics table.
"""

from pathlib import Path
import sqlite3

import pandas as pd


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

# Current file:
# CO_ADOPTAI/src/merge_db.py
#
# parent       -> src/
# parent.parent -> CO_ADOPTAI/
BASE_DIR = Path(__file__).resolve().parent.parent

# Folder containing the original databases.
INPUT_DIR = BASE_DIR / "data" / "raw" / "db"

# Folder where the merged files will be created.
OUTPUT_DIR = BASE_DIR / "data" / "merged"

# Final merged SQLite database.
OUTPUT_DB = OUTPUT_DIR / "merged_metrics.db"

# CSV exported from the merged system_metrics table.
OUTPUT_CSV = OUTPUT_DIR / "merged_metrics.csv"

# ---------------------------------------------------------
# TABLE CONFIGURATION
# ---------------------------------------------------------

# Preferred order when writing the tables into the final DB.
TABLE_ORDER = [
    "schema_meta",
    "runs",
    "events",
    "capabilities",
    "system_metrics",
]


# Columns used to recognize duplicate rows in each table.
DEDUP_KEYS = {
    # One metadata value for each key.
    "schema_meta": ["key"],

    # Each collection session has one unique run_id.
    "runs": ["run_id"],

    # Same run, time, event type and message means the same event.
    "events": [
        "run_id",
        "timestamp_utc",
        "event_type",
        "message",
    ],

    # One capability status for each metric in one run.
    "capabilities": [
        "run_id",
        "metric_name",
    ],

    # One measurement for the same machine, run and timestamp.
    "system_metrics": [
        "machine_id",
        "run_id",
        "timestamp",
    ],
}


# These ID columns may have the same values in different databases.
# They will therefore be regenerated after merging.
AUTOINCREMENT_COLS = {
    "events": "id",
    "system_metrics": "id",
}


def get_table_names(connection: sqlite3.Connection) -> list[str]:
    """
    Return all user-created tables from a SQLite database.

    SQLite internal tables such as sqlite_sequence are ignored.
    """

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
          AND name NOT LIKE 'sqlite_%'
        """
    )

    return [row[0] for row in cursor.fetchall()]


def main() -> None:
    """
    Main merge process.

    Steps:
    1. Find all .db files.
    2. Read every table from every database.
    3. Merge tables with the same name.
    4. Remove duplicates.
    5. Sort the data.
    6. Regenerate conflicting IDs.
    7. Create the final DB.
    8. Export system_metrics to CSV.
    """

    # Create folders automatically if they do not exist.
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find all SQLite databases inside data/raw/db.
    db_files = sorted(INPUT_DIR.glob("*.db"))

    # Safety: exclude the output DB if it was accidentally placed
    # inside the input folder.
    db_files = [
        db_file
        for db_file in db_files
        if db_file.resolve() != OUTPUT_DB.resolve()
    ]

    if not db_files:
        raise SystemExit(
            f"No SQLite database found in:\n{INPUT_DIR}"
        )

    # Structure:
    #
    # {
    #     "runs": [dataframe1, dataframe2],
    #     "system_metrics": [dataframe1, dataframe2]
    # }
    tables_data: dict[str, list[pd.DataFrame]] = {}

    # -----------------------------------------------------
    # READ ALL SOURCE DATABASES
    # -----------------------------------------------------

    for db_file in db_files:
        try:
            # Open the source database in read-only mode.
            # The original database cannot be modified.
            database_uri = db_file.resolve().as_uri() + "?mode=ro"

            with sqlite3.connect(
                database_uri,
                uri=True,
            ) as connection:

                table_names = get_table_names(connection)

                # Read every table from this database.
                for table_name in table_names:
                    query = f'SELECT * FROM "{table_name}"'

                    dataframe = pd.read_sql_query(
                        query,
                        connection,
                    )

                    # Ignore empty tables.
                    if not dataframe.empty:
                        tables_data.setdefault(
                            table_name,
                            [],
                        ).append(dataframe)

            print(
                f"✓ {db_file.name}: "
                f"tables found = {table_names}"
            )

        except Exception as error:
            # One invalid database does not stop the other databases.
            print(
                f"✗ Error reading {db_file.name}: {error}"
            )

    if not tables_data:
        raise SystemExit(
            "No valid data was found in the source databases."
        )

    # Remove the old merged DB before recreating it.
    if OUTPUT_DB.exists():
        OUTPUT_DB.unlink()

    # Stores the final merged DataFrames.
    merged_tables: dict[str, pd.DataFrame] = {}

    # First use the expected table order.
    ordered_table_names = [
        table_name
        for table_name in TABLE_ORDER
        if table_name in tables_data
    ]

    # Add unexpected tables at the end.
    ordered_table_names += [
        table_name
        for table_name in tables_data
        if table_name not in ordered_table_names
    ]

    # -----------------------------------------------------
    # MERGE AND WRITE TABLES
    # -----------------------------------------------------

    try:
        with sqlite3.connect(OUTPUT_DB) as output_connection:

            for table_name in ordered_table_names:
                dataframes = tables_data[table_name]

                # Put the rows from all databases one after another.
                merged_dataframe = pd.concat(
                    dataframes,
                    ignore_index=True,
                    sort=False,
                )

                rows_before = len(merged_dataframe)

                # Get the duplicate identification columns.
                dedup_columns = DEDUP_KEYS.get(table_name)

                # Remove duplicates using the configured columns.
                if (
                    dedup_columns
                    and set(dedup_columns).issubset(
                        merged_dataframe.columns
                    )
                ):
                    merged_dataframe.drop_duplicates(
                        subset=dedup_columns,
                        inplace=True,
                    )
                else:
                    # If no key is configured, remove fully identical rows.
                    merged_dataframe.drop_duplicates(
                        inplace=True
                    )

                removed_rows = (
                    rows_before - len(merged_dataframe)
                )

                # Sort the table using its deduplication columns.
                sort_columns = [
                    column
                    for column in (dedup_columns or [])
                    if column in merged_dataframe.columns
                ]

                if sort_columns:
                    merged_dataframe.sort_values(
                        by=sort_columns,
                        inplace=True,
                    )

                # Recreate the pandas index:
                # 0, 1, 2, 3, ...
                merged_dataframe.reset_index(
                    drop=True,
                    inplace=True,
                )

                # Regenerate SQL IDs to avoid collisions.
                #
                # Example before:
                # DB 1 IDs: 1, 2, 3
                # DB 2 IDs: 1, 2, 3
                #
                # After:
                # 1, 2, 3, 4, 5, 6
                id_column = AUTOINCREMENT_COLS.get(table_name)

                if (
                    id_column
                    and id_column in merged_dataframe.columns
                ):
                    merged_dataframe[id_column] = range(
                        1,
                        len(merged_dataframe) + 1,
                    )

                # Store the final DataFrame in memory.
                merged_tables[table_name] = merged_dataframe

                # Write the table into the final SQLite database.
                merged_dataframe.to_sql(
                    table_name,
                    output_connection,
                    index=False,
                    if_exists="replace",
                )

                print(
                    f"  - {table_name}: "
                    f"{len(merged_dataframe)} rows, "
                    f"{removed_rows} duplicates removed"
                )

            # -------------------------------------------------
            # CREATE DATABASE INDEXES
            # -------------------------------------------------

            cursor = output_connection.cursor()

            # Faster filtering by machine and time.
            if "system_metrics" in merged_tables:
                columns = merged_tables[
                    "system_metrics"
                ].columns

                if {
                    "machine_id",
                    "timestamp",
                }.issubset(columns):
                    cursor.execute(
                        """
                        CREATE INDEX IF NOT EXISTS
                        idx_machine_timestamp
                        ON system_metrics(
                            machine_id,
                            timestamp
                        )
                        """
                    )

            # Faster filtering of events by run.
            if (
                "events" in merged_tables
                and "run_id"
                in merged_tables["events"].columns
            ):
                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS
                    idx_events_run
                    ON events(run_id)
                    """
                )

            # Faster filtering of capabilities by run.
            if (
                "capabilities" in merged_tables
                and "run_id"
                in merged_tables["capabilities"].columns
            ):
                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS
                    idx_capabilities_run
                    ON capabilities(run_id)
                    """
                )

            # Save database changes.
            output_connection.commit()

    except Exception as error:
        # Delete incomplete output if the merge fails.
        if OUTPUT_DB.exists():
            OUTPUT_DB.unlink()

        raise SystemExit(
            f"Error creating the merged database: {error}"
        )

    # -----------------------------------------------------
    # EXPORT SYSTEM_METRICS TO CSV
    # -----------------------------------------------------

    if "system_metrics" not in merged_tables:
        raise SystemExit(
            "The databases were merged, but the "
            "system_metrics table was not found."
        )

    merged_tables["system_metrics"].to_csv(
        OUTPUT_CSV,
        index=False,
    )

    # -----------------------------------------------------
    # FINAL REPORT
    # -----------------------------------------------------

    print("\nMerge completed successfully.")
    print(f"Databases processed: {len(db_files)}")
    print(
        f"Tables merged: {list(merged_tables.keys())}"
    )
    print(f"Merged DB:  {OUTPUT_DB}")
    print(f"Merged CSV: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()