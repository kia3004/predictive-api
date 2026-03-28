from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import importlib.util
import os

import mysql.connector


BATCH_PREFIX = os.getenv("VERIFY_PRED_BATCH_PREFIX", "VERIFYPRED-20260328")
START_DATE = date(2025, 11, 27)
BARANGAYS = ["Sevilla", "Catbangen", "San Vicente"]
DISEASES = ["Dengue", "Influenza", "Leptospirosis"]
BASELINE_DAY_INDICES = list(range(0, 68, 2))
RECENT_DAY_INDICES = [100, 106, 112, 118]
ACTIVE_DAY_INDICES = BASELINE_DAY_INDICES + RECENT_DAY_INDICES
SYMPTOMS = {
    "Dengue": "Synthetic verification batch: fever, headache, retro-orbital pain",
    "Influenza": "Synthetic verification batch: cough, fever, sore throat",
    "Leptospirosis": "Synthetic verification batch: fever, myalgia, headache",
}
BOOST_CASES = {
    # Create clear late-series leaders so the CHO dashboard has an obvious
    # winning barangay for each disease after the predictor refresh.
    ("Sevilla", "Dengue"): {100: 4, 106: 5, 112: 7, 116: 9, 118: 12},
    ("Catbangen", "Influenza"): {100: 4, 106: 6, 112: 8, 116: 10, 118: 14},
    ("San Vicente", "Leptospirosis"): {100: 5, 106: 7, 112: 9, 116: 12, 118: 15},
}


def get_connection():
    return mysql.connector.connect(
        host=os.getenv("MYSQL_HOST", "centerbeam.proxy.rlwy.net"),
        port=int(os.getenv("MYSQL_PORT", "38661")),
        user=os.getenv("MYSQL_USER", "phimas_user"),
        password=os.getenv("MYSQL_PASSWORD", "Phimas123!"),
        database=os.getenv("MYSQL_DATABASE", "railway"),
    )


def load_prediction_module():
    module_path = Path(__file__).resolve().parents[1] / "main.py"
    spec = importlib.util.spec_from_file_location("prediction_api_main", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def build_baseline_case_count(barangay_index: int, disease_index: int, active_order: int):
    return 1 + ((active_order + disease_index + barangay_index) % 2)


def build_status(day_index: int):
    return "Done" if day_index < 90 else "Monitoring"


def load_reference_data(cursor):
    cursor.execute("SELECT UserID, AssignedArea FROM users WHERE Role = 'BHW'")
    bhw_by_area = {row["AssignedArea"]: row["UserID"] for row in cursor.fetchall()}

    cursor.execute(
        """
        SELECT hm.PatientID, h.Address
        FROM household_members hm
        INNER JOIN households h ON hm.HouseholdID = h.HouseholdID
        WHERE hm.IsEmergencyContact = 0
        ORDER BY hm.HouseholdID, hm.PatientID
        """
    )

    patients_by_barangay = {barangay: [] for barangay in BARANGAYS}
    for row in cursor.fetchall():
        barangay = row["Address"].split(",", 1)[0].strip()
        if barangay in patients_by_barangay:
            patients_by_barangay[barangay].append(row["PatientID"])

    return bhw_by_area, patients_by_barangay


def load_existing_markers(cursor):
    cursor.execute(
        "SELECT Symptoms FROM health_records WHERE Symptoms LIKE %s",
        (f"{BATCH_PREFIX}|%",),
    )
    return {row["Symptoms"] for row in cursor.fetchall()}


def append_record(records_to_insert, existing_markers, patient_pool, bhw_id, barangay, disease, day_index, case_index, stage):
    marker = f"{BATCH_PREFIX}|{stage}|{barangay}|{disease}|day{day_index:03d}|case{case_index:02d}"
    if marker in existing_markers:
        return

    patient_id = patient_pool[(day_index + case_index) % len(patient_pool)]
    record_date = START_DATE + timedelta(days=day_index)
    records_to_insert.append(
        (
            patient_id,
            bhw_id,
            record_date,
            disease,
            marker,
            build_status(day_index),
        )
    )
    existing_markers.add(marker)


def insert_batch_records(cursor):
    bhw_by_area, patients_by_barangay = load_reference_data(cursor)
    existing_markers = load_existing_markers(cursor)
    records_to_insert = []

    for barangay_index, barangay in enumerate(BARANGAYS):
        patient_pool = patients_by_barangay[barangay]
        bhw_id = bhw_by_area[barangay]

        for disease_index, disease in enumerate(DISEASES):
            for active_order, day_index in enumerate(ACTIVE_DAY_INDICES):
                baseline_cases = build_baseline_case_count(barangay_index, disease_index, active_order)
                for case_index in range(1, baseline_cases + 1):
                    append_record(
                        records_to_insert,
                        existing_markers,
                        patient_pool,
                        bhw_id,
                        barangay,
                        disease,
                        day_index,
                        case_index,
                        "baseline",
                    )

            extra_boosts = BOOST_CASES.get((barangay, disease), {})
            for day_index, boost_cases in extra_boosts.items():
                for case_index in range(1, boost_cases + 1):
                    append_record(
                        records_to_insert,
                        existing_markers,
                        patient_pool,
                        bhw_id,
                        barangay,
                        disease,
                        day_index,
                        case_index,
                        "boost",
                    )

    if records_to_insert:
        cursor.executemany(
            """
            INSERT INTO health_records
                (PatientID, BHWID, DateRecorded, Disease, Symptoms, Status)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            records_to_insert,
        )

    return len(records_to_insert)


def refresh_saved_predictions():
    module = load_prediction_module()
    saved_rows = []

    for barangay in BARANGAYS:
        for disease in DISEASES:
            data = module.get_data(barangay_name=barangay, disease_name=disease)
            result = module.predict_barangay_disease(data, barangay, disease)
            if "error" in result:
                saved_rows.append({"barangay": barangay, "disease": disease, "error": result["error"]})
                continue

            save_result = module.save_prediction_to_db(result)
            saved_rows.append(
                {
                    "barangay": barangay,
                    "disease": disease,
                    "predicted_cases": result["predicted_cases"],
                    "confidence": result["confidence_ratio"],
                    "save_status": save_result["status"],
                }
            )

    return saved_rows


def print_latest_winners(cursor):
    cursor.execute(
        """
        SELECT Disease, HighRiskBarangay, PredictedCases, ConfidenceScore, DateGenerated
        FROM predictive_analysis
        WHERE DATE(DateGenerated) = CURDATE()
          AND Disease IN (%s, %s, %s)
          AND HighRiskBarangay IN (%s, %s, %s)
        ORDER BY Disease, PredictedCases DESC, ConfidenceScore DESC
        """,
        (*DISEASES, *BARANGAYS),
    )

    print("\nLatest saved prediction ranking for today:")
    for row in cursor.fetchall():
        print(row)


def main():
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    try:
        inserted_count = insert_batch_records(cursor)
        conn.commit()
        saved_rows = refresh_saved_predictions()

        print(f"Inserted verification records: {inserted_count}")
        print("Saved prediction refresh:")
        for row in saved_rows:
            print(row)

        print_latest_winners(cursor)
    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    main()
