from datetime import datetime
from fastapi import FastAPI, HTTPException
import mysql.connector
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

app = FastAPI()

# =========================================================
# DATABASE
# =========================================================
def get_connection():
    return mysql.connector.connect(
        host="centerbeam.proxy.rlwy.net",
        port=38661,
        user="phimas_user",
        password="Phimas123!",
        database="railway",
    )

# =========================================================
# LOAD DATA
# =========================================================
def get_data():
    conn = get_connection()

    query = """
        SELECT 
            DATE(hr.DateRecorded) as DateRecorded,
            hr.Disease,
            COALESCE(h.Address, 'Unknown') as Address,
            COUNT(*) as Cases
        FROM health_records hr
        LEFT JOIN households h ON hr.HouseholdID = h.HouseholdID
        GROUP BY DateRecorded, hr.Disease, Address
        ORDER BY DateRecorded
    """

    df = pd.read_sql(query, conn)
    conn.close()

    df["DateRecorded"] = pd.to_datetime(df["DateRecorded"])
    df["Barangay"] = df["Address"].str.split(",").str[0]

    return df

# =========================================================
# BUILD ML DATASET
# =========================================================
def build_ml_dataset(data):
    df = data.copy()

    df = df.sort_values(["Disease","Barangay","DateRecorded"])

    g = df.groupby(["Disease","Barangay"])

    df["Lag1"] = g["Cases"].shift(1)
    df["Lag2"] = g["Cases"].shift(2)
    df["Lag3"] = g["Cases"].shift(3)

    df["Target"] = g["Cases"].shift(-1)

    df["Month"] = df["DateRecorded"].dt.month
    df["Day"] = df["DateRecorded"].dt.day
    df["DayOfWeek"] = df["DateRecorded"].dt.dayofweek

    df = df.dropna()

    return df

# =========================================================
# TRAIN MODEL
# =========================================================
def train_model():
    data = get_data()
    dataset = build_ml_dataset(data)

    X = dataset[[
        "Disease","Barangay",
        "Lag1","Lag2","Lag3",
        "Month","Day","DayOfWeek"
    ]]
    
    y = dataset["Target"]

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown='ignore'), ["Disease","Barangay"]),
        ("num", StandardScaler(), ["Lag1","Lag2","Lag3","Month","Day","DayOfWeek"])
    ])

    model = Pipeline([
        ("prep", preprocessor),
        ("model", RandomForestRegressor(n_estimators=100))
    ])

    model.fit(X, y)
    return model

# =========================================================
# LOAD MODEL ON STARTUP
# =========================================================
print("Training ML model...")
ml_model = train_model()
print("Model ready!")

# =========================================================
# PREDICTION FUNCTION
# =========================================================
def predict_cases(ts, disease_name, barangay_name=None, forecast_days=7):

    if ts.empty:
        return {"error": "No data"}

    forecast_values = []

    working = ts.reset_index()
    working.columns = ["DateRecorded","Cases"]

    for i in range(forecast_days):
        last = working.tail(3)

        if len(last) < 3:
            return {"error": "Not enough data"}

        lag1 = last.iloc[-1]["Cases"]
        lag2 = last.iloc[-2]["Cases"]
        lag3 = last.iloc[-3]["Cases"]

        new_data = pd.DataFrame([{
            "Disease": disease_name,
            "Barangay": barangay_name if barangay_name else "Unknown",
            "Lag1": lag1,
            "Lag2": lag2,
            "Lag3": lag3,
            "Month": datetime.now().month,
            "Day": datetime.now().day,
            "DayOfWeek": datetime.now().weekday()
        }])

        pred = ml_model.predict(new_data)[0]
        pred = max(0, round(pred, 2))

        forecast_values.append(pred)

        working = pd.concat([
            working,
            pd.DataFrame([{
                "DateRecorded": datetime.now(),
                "Cases": pred
            }])
        ])

    return forecast_values

# =========================================================
# API ENDPOINT
# =========================================================
@app.get("/predict")
def predict(disease: str, barangay: str = None):

    data = get_data()

    if barangay:
        filtered = data[
            (data["Disease"].str.lower() == disease.lower()) &
            (data["Barangay"].str.lower() == barangay.lower())
        ]
    else:
        filtered = data[
            data["Disease"].str.lower() == disease.lower()
        ]

    if filtered.empty:
        raise HTTPException(status_code=404, detail="No data found")

    ts = filtered.groupby("DateRecorded")["Cases"].sum()

    forecast = predict_cases(ts, disease, barangay)

    return {
        "disease": disease,
        "barangay": barangay,
        "forecast": forecast
    }