from __future__ import annotations

import argparse
import os

import mysql.connector


DEFAULT_BATCH_PREFIX = os.getenv("VERIFY_PRED_BATCH_PREFIX", "VERIFYPRED-20260328")
BARANGAYS = ["Sevilla", "Catbangen", "San Vicente"]
DISEASES = ["Dengue", "Influenza", "Leptospirosis"]


def get_connection():
    return mysql.connector.connect(
        host=os.getenv("MYSQL_HOST", "centerbeam.proxy.rlwy.net"),
        port=int(os.getenv("MYSQL_PORT", "38661")),
        user=os.getenv("MYSQL_USER", "phimas_user"),
        password=os.getenv("MYSQL_PASSWORD", "Phimas123!"),
        database=os.getenv("MYSQL_DATABASE", "railway"),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Remove only the verification-marked health-record batch.")
    parser.add_argument("--batch-prefix", default=DEFAULT_BATCH_PREFIX)
    parser.add_argument(
        "--delete-predictions-for-date",
        metavar="YYYY-MM-DD",
        help="Optionally delete saved predictive_analysis rows for the verification disease/barangay pairs on a specific date.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            "SELECT COUNT(*) FROM health_records WHERE Symptoms LIKE %s",
            (f"{args.batch_prefix}|%",),
        )
        record_count = cursor.fetchone()[0]

        cursor.execute(
            "DELETE FROM health_records WHERE Symptoms LIKE %s",
            (f"{args.batch_prefix}|%",),
        )
        deleted_records = cursor.rowcount
        deleted_predictions = 0

        if args.delete_predictions_for_date:
            cursor.execute(
                """
                DELETE FROM predictive_analysis
                WHERE DATE(DateGenerated) = %s
                  AND Disease IN (%s, %s, %s)
                  AND HighRiskBarangay IN (%s, %s, %s)
                """,
                (args.delete_predictions_for_date, *DISEASES, *BARANGAYS),
            )
            deleted_predictions = cursor.rowcount

        conn.commit()

        print(f"Matched health_records: {record_count}")
        print(f"Deleted health_records: {deleted_records}")
        if args.delete_predictions_for_date:
            print(f"Deleted predictive_analysis rows on {args.delete_predictions_for_date}: {deleted_predictions}")
        else:
            print("Predictive analysis rows were left untouched.")
    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    main()
