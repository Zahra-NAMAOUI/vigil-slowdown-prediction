import sqlite3
import csv
import os

DB_PATH = os.path.join("data", "metrics.db")
CSV_PATH = os.path.join("data", "dataset.csv")
TABLE_NAME = "system_metrics_v2"   # change en "measurements" si tu utilises encore la v1


def export_to_csv():
    if not os.path.exists(DB_PATH):
        print(f"[!] Base introuvable : {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(f"SELECT * FROM {TABLE_NAME}")
    rows = cursor.fetchall()
    column_names = [description[0] for description in cursor.description]

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(column_names)   # ligne d'en-tête
        writer.writerows(rows)          # toutes les données

    conn.close()
    print(f"[+] Export terminé : {len(rows)} lignes écrites dans '{CSV_PATH}'")


if __name__ == "__main__":
    export_to_csv()