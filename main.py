from datetime import datetime
from io import BytesIO
import os
import re

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
import mysql.connector
import numpy as np
import pandas as pd
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA

app = FastAPI()


def get_connection():
    return mysql.connector.connect(
        host=os.getenv("MYSQL_HOST", "centerbeam.proxy.rlwy.net"),
        port=int(os.getenv("MYSQL_PORT", "38661")),
        user=os.getenv("MYSQL_USER", "phimas_user"),
        password=os.getenv("MYSQL_PASSWORD", "Phimas123!"),
        database=os.getenv("MYSQL_DATABASE", "railway"),
    )


def get_existing_table_name(conn, candidates):
    cursor = conn.cursor()

    try:
        for table_name in candidates:
            cursor.execute("SHOW TABLES LIKE %s", (table_name,))
            if cursor.fetchone():
                return table_name
    finally:
        cursor.close()

    return None


def get_table_columns(conn, table_name):
    cursor = conn.cursor(dictionary=True)

    try:
        cursor.execute(f"SHOW COLUMNS FROM `{table_name}`")
        return {row["Field"] for row in cursor.fetchall()}
    finally:
        cursor.close()


def build_barangay_expression(household_columns):
    if "Barangay" in household_columns:
        return "COALESCE(NULLIF(TRIM(h.Barangay), ''), NULLIF(TRIM(SUBSTRING_INDEX(h.Address, ',', 1)), ''), 'Unknown')"

    if "Address" in household_columns:
        return "COALESCE(NULLIF(TRIM(SUBSTRING_INDEX(h.Address, ',', 1)), ''), 'Unknown')"

    return "'Unknown'"


def get_data(barangay_name: str | None = None, disease_name: str | None = None):
    conn = get_connection()

    try:
        health_table = get_existing_table_name(conn, ["healthrecords", "health_records"])
        household_table = get_existing_table_name(conn, ["households", "household"])

        if not health_table or not household_table:
            raise HTTPException(status_code=500, detail="Required health data tables were not found.")

        household_columns = get_table_columns(conn, household_table)
        barangay_expression = build_barangay_expression(household_columns)

        query = f"""
            SELECT
                DATE(hr.DateRecorded) AS DateRecorded,
                TRIM(hr.Disease) AS Disease,
                {barangay_expression} AS Barangay,
                COUNT(*) AS Cases
            FROM `{health_table}` hr
            INNER JOIN `{household_table}` h
                ON hr.HouseholdID = h.HouseholdID
            WHERE TRIM(hr.Disease) <> ''
        """

        params = []

        if disease_name:
            query += " AND LOWER(TRIM(hr.Disease)) = LOWER(%s)"
            params.append(disease_name.strip())

        if barangay_name:
            query += f" AND LOWER({barangay_expression}) = LOWER(%s)"
            params.append(barangay_name.strip())

        query += f"""
            GROUP BY
                DATE(hr.DateRecorded),
                TRIM(hr.Disease),
                {barangay_expression}
            ORDER BY
                DATE(hr.DateRecorded),
                TRIM(hr.Disease),
                {barangay_expression}
        """

        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute(query, params)
            rows = cursor.fetchall()
        finally:
            cursor.close()
    finally:
        conn.close()

    if not rows:
        return pd.DataFrame(columns=["DateRecorded", "Disease", "Barangay", "Cases"])

    data = pd.DataFrame(rows)
    data["DateRecorded"] = pd.to_datetime(data["DateRecorded"])
    data["Cases"] = pd.to_numeric(data["Cases"], errors="coerce").fillna(0)
    data["Disease"] = data["Disease"].astype(str).str.strip()
    data["Barangay"] = data["Barangay"].astype(str).str.strip()

    return data


def build_time_series(filtered_data: pd.DataFrame):
    daily_cases = (
        filtered_data.groupby("DateRecorded", as_index=True)["Cases"]
        .sum()
        .sort_index()
    )

    daily_cases.index = pd.DatetimeIndex(daily_cases.index)
    return daily_cases.asfreq("D").fillna(0)


def build_prediction_result(ts: pd.Series, disease_name: str, barangay_name: str | None = None, forecast_days: int = 7):
    if ts.empty:
        return {"error": "No data found for this request"}

    if len(ts) < 10:
        return {"error": "Not enough data for prediction"}

    train_size = int(len(ts) * 0.8)
    train = ts.iloc[:train_size]
    test = ts.iloc[train_size:]

    if train.empty or test.empty:
        return {"error": "Not enough data for prediction"}

    try:
        model_fit = ARIMA(train, order=(2, 1, 2)).fit()
        test_forecast = model_fit.forecast(steps=len(test))

        mae = mean_absolute_error(test, test_forecast)
        rmse = float(np.sqrt(((test - test_forecast) ** 2).mean()))

        final_model = ARIMA(ts, order=(2, 1, 2)).fit()
        forecast = final_model.forecast(steps=forecast_days)
    except Exception as exc:
        return {"error": f"Prediction failed: {exc}"}

    forecast = pd.Series(np.maximum(forecast, 0), index=forecast.index)
    forecast_values = [round(float(value), 2) for value in forecast.tolist()]
    forecast_dates = [index.strftime("%Y-%m-%d") for index in forecast.index]

    average_cases = float(ts.mean())

    def risk_score(value):
        if value > average_cases * 1.5:
            return "High"
        if value > average_cases:
            return "Medium"
        return "Low"

    risk_levels = [risk_score(value) for value in forecast_values]
    risk_priority = {"Low": 1, "Medium": 2, "High": 3}
    overall_risk_level = max(risk_levels, key=lambda level: risk_priority[level], default="Low")

    first_value = forecast_values[0]
    last_value = forecast_values[-1]
    if last_value > first_value:
        trend = "Increasing"
    elif last_value < first_value:
        trend = "Decreasing"
    else:
        trend = "Stable"

    confidence_percent = max(0.0, min(100.0, 100 - (rmse * 5)))
    confidence_ratio = round(confidence_percent / 100, 4)

    result = {
        "disease": disease_name,
        "forecast": forecast_values,
        "forecast_dates": forecast_dates,
        "risk_levels": risk_levels,
        "overall_risk_level": overall_risk_level,
        "trend": trend,
        "predicted_cases": int(round(sum(forecast_values))),
        "confidence_score": round(confidence_percent, 2),
        "confidence_ratio": confidence_ratio,
        "mae": round(float(mae), 2),
        "rmse": round(rmse, 2),
    }

    if barangay_name:
        result["barangay"] = barangay_name

    return result


def predict_disease(data: pd.DataFrame, disease_name: str, forecast_days: int = 7):
    if data.empty:
        return {"error": "No data found for this disease"}

    disease_key = disease_name.strip().casefold()
    disease_data = data[data["Disease"].str.casefold() == disease_key].copy()

    if disease_data.empty:
        return {"error": "No data found for this disease"}

    series = build_time_series(disease_data)
    matched_disease = disease_data["Disease"].iloc[0]
    return build_prediction_result(series, matched_disease, forecast_days=forecast_days)


def predict_barangay_disease(data: pd.DataFrame, barangay_name: str, disease_name: str, forecast_days: int = 7):
    if data.empty:
        return {"error": "No data found for this barangay and disease"}

    barangay_key = barangay_name.strip().casefold()
    disease_key = disease_name.strip().casefold()

    filtered_data = data[
        (data["Barangay"].str.casefold() == barangay_key)
        & (data["Disease"].str.casefold() == disease_key)
    ].copy()

    if filtered_data.empty:
        return {"error": "No data found for this barangay and disease"}

    series = build_time_series(filtered_data)
    matched_barangay = filtered_data["Barangay"].iloc[0]
    matched_disease = filtered_data["Disease"].iloc[0]
    return build_prediction_result(series, matched_disease, matched_barangay, forecast_days)


def save_prediction_to_db(prediction_result):
    conn = get_connection()

    try:
        table_name = get_existing_table_name(
            conn,
            ["predictiveanalysis", "predictiveanalyses", "predictive_analysis"],
        )

        if not table_name:
            raise HTTPException(status_code=500, detail="Predictive analysis table was not found.")

        columns = get_table_columns(conn, table_name)
        location_column = "Barangay" if "Barangay" in columns else "HighRiskBarangay" if "HighRiskBarangay" in columns else None

        if not location_column:
            raise HTTPException(status_code=500, detail="No barangay column exists in the predictive analysis table.")

        values = {
            "DateGenerated": datetime.utcnow(),
            "Disease": prediction_result["disease"],
            "PredictedCases": prediction_result["predicted_cases"],
            "ConfidenceScore": prediction_result["confidence_ratio"],
            "RiskLevel": prediction_result["overall_risk_level"],
            "Trend": prediction_result["trend"],
            "Barangay": prediction_result.get("barangay", "Unknown"),
            "HighRiskBarangay": prediction_result.get("barangay", "Unknown"),
        }

        insert_columns = [column for column in values if column in columns]

        if not insert_columns:
            raise HTTPException(status_code=500, detail="No compatible predictive analysis columns were found.")

        cursor = conn.cursor(dictionary=True)

        try:
            cursor.execute(
                f"""
                    SELECT 1
                    FROM `{table_name}`
                    WHERE DATE(DateGenerated) = %s
                      AND LOWER(Disease) = LOWER(%s)
                      AND LOWER(`{location_column}`) = LOWER(%s)
                    LIMIT 1
                """,
                (
                    values["DateGenerated"].date(),
                    values["Disease"],
                    values[location_column],
                ),
            )
            record_exists = cursor.fetchone() is not None

            if record_exists:
                set_clause = ", ".join(f"`{column}` = %s" for column in insert_columns)
                update_values = [values[column] for column in insert_columns]
                update_values.extend(
                    [
                        values["DateGenerated"].date(),
                        values["Disease"],
                        values[location_column],
                    ]
                )

                cursor.execute(
                    f"""
                        UPDATE `{table_name}`
                        SET {set_clause}
                        WHERE DATE(DateGenerated) = %s
                          AND LOWER(Disease) = LOWER(%s)
                          AND LOWER(`{location_column}`) = LOWER(%s)
                    """,
                    update_values,
                )
                action = "updated"
            else:
                placeholders = ", ".join(["%s"] * len(insert_columns))
                column_clause = ", ".join(f"`{column}`" for column in insert_columns)
                insert_values = [values[column] for column in insert_columns]

                cursor.execute(
                    f"INSERT INTO `{table_name}` ({column_clause}) VALUES ({placeholders})",
                    insert_values,
                )
                action = "saved"

            conn.commit()
        finally:
            cursor.close()
    finally:
        conn.close()

    return {"status": action, "table": table_name}


def generate_pdf(prediction_result):
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=LETTER)

    _, height = LETTER
    y = height - 50

    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(50, y, "PHIMAS Predictive Analysis Report")
    y -= 30

    pdf.setFont("Helvetica", 11)
    summary_lines = [
        f"Date Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        f"Disease: {prediction_result['disease']}",
        f"Barangay: {prediction_result.get('barangay', 'All Barangays')}",
        f"Predicted Cases (next 7 days): {prediction_result['predicted_cases']}",
        f"Trend: {prediction_result['trend']}",
        f"Overall Risk Level: {prediction_result['overall_risk_level']}",
        f"Confidence Score: {prediction_result['confidence_score']}%",
    ]

    for line in summary_lines:
        pdf.drawString(50, y, line)
        y -= 18

    y -= 10
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, y, "Forecast Details")
    y -= 20
    pdf.setFont("Helvetica", 11)

    for forecast_date, value, risk in zip(
        prediction_result["forecast_dates"],
        prediction_result["forecast"],
        prediction_result["risk_levels"],
    ):
        pdf.drawString(50, y, f"{forecast_date}: {value} cases | Risk: {risk}")
        y -= 18

        if y < 50:
            pdf.showPage()
            y = height - 50
            pdf.setFont("Helvetica", 11)

    pdf.save()
    buffer.seek(0)
    return buffer


def sanitize_filename(value: str):
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip())
    return cleaned.strip("_") or "report"


@app.get("/")
def root():
    return {"message": "PHIMAS Predictive API is running"}


@app.get("/predict/{disease}")
def predict(disease: str, forecast_days: int = Query(default=7, ge=1, le=30)):
    data = get_data(disease_name=disease)
    return predict_disease(data, disease, forecast_days)


@app.get("/predict/{barangay}/{disease}")
def predict_by_barangay(barangay: str, disease: str, forecast_days: int = Query(default=7, ge=1, le=30)):
    data = get_data(barangay_name=barangay, disease_name=disease)
    return predict_barangay_disease(data, barangay, disease, forecast_days)


@app.get("/predict_all")
def predict_all(forecast_days: int = Query(default=7, ge=1, le=30)):
    data = get_data()

    if data.empty:
        return []

    diseases = sorted(data["Disease"].dropna().unique())
    results = []

    for disease in diseases:
        result = predict_disease(data, disease, forecast_days)
        if "error" not in result:
            results.append(result)

    return results


@app.get("/predict-and-save/{barangay}/{disease}")
def predict_and_save(barangay: str, disease: str, forecast_days: int = Query(default=7, ge=1, le=30)):
    data = get_data(barangay_name=barangay, disease_name=disease)
    result = predict_barangay_disease(data, barangay, disease, forecast_days)

    if "error" in result:
        return result

    save_result = save_prediction_to_db(result)
    result["save_status"] = save_result["status"]
    result["saved_to"] = save_result["table"]

    return result


@app.get("/report/{barangay}/{disease}")
def report(barangay: str, disease: str, forecast_days: int = Query(default=7, ge=1, le=30)):
    data = get_data(barangay_name=barangay, disease_name=disease)
    result = predict_barangay_disease(data, barangay, disease, forecast_days)

    if "error" in result:
        return result

    pdf_buffer = generate_pdf(result)
    safe_barangay = sanitize_filename(result.get("barangay", barangay))
    safe_disease = sanitize_filename(result["disease"])
    filename = f"{safe_barangay}_{safe_disease}_report.pdf"

    return StreamingResponse(
        pdf_buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
