from contextlib import asynccontextmanager
from datetime import timedelta
from functools import lru_cache
import os

from fastapi import FastAPI, HTTPException, Request
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# =========================================================
# DATABASE
# =========================================================
def get_env(*names, default=None):
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


def normalize_database_url(database_url):
    if database_url.startswith("mysql://"):
        return database_url.replace("mysql://", "mysql+mysqlconnector://", 1)

    if database_url.startswith("mariadb://"):
        return database_url.replace("mariadb://", "mysql+mysqlconnector://", 1)

    return database_url


@lru_cache(maxsize=1)
def get_engine():
    database_url = get_env("DATABASE_URL", "MYSQL_URL")
    if database_url:
        return create_engine(normalize_database_url(database_url), pool_pre_ping=True)

    host = get_env("MYSQLHOST", "MYSQL_HOST", "DB_HOST", default="centerbeam.proxy.rlwy.net")
    port = int(get_env("MYSQLPORT", "MYSQL_PORT", "DB_PORT", default="38661"))
    user = get_env("MYSQLUSER", "MYSQL_USER", "DB_USER", default="phimas_user")
    password = get_env("MYSQLPASSWORD", "MYSQL_PASSWORD", "DB_PASSWORD", default="Phimas123!")
    database = get_env("MYSQLDATABASE", "MYSQL_DATABASE", "DB_NAME", default="railway")

    return create_engine(
        URL.create(
            "mysql+mysqlconnector",
            username=user,
            password=password,
            host=host,
            port=port,
            database=database,
        ),
        pool_pre_ping=True,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ml_model = None
    app.state.model_error = None

    try:
        print("Training ML model...")
        app.state.ml_model = train_model()
        print("Model ready!")
    except Exception as exc:
        app.state.model_error = str(exc)
        print(f"Model training failed: {exc}")

    yield

    get_engine().dispose()


app = FastAPI(lifespan=lifespan)

# =========================================================
# LOAD DATA
# =========================================================
def get_data():
    query = """
        SELECT 
            DATE(hr.DateRecorded) AS DateRecorded,
            hr.Disease,
            COALESCE(NULLIF(TRIM(SUBSTRING_INDEX(h.Address, ',', 1)), ''), 'Unknown') AS Barangay,
            COUNT(*) AS Cases
        FROM health_records hr
        INNER JOIN household_members hm ON hr.PatientID = hm.PatientID
        INNER JOIN households h ON hm.HouseholdID = h.HouseholdID
        WHERE hr.DateRecorded IS NOT NULL
          AND hr.Disease IS NOT NULL
        GROUP BY
            DATE(hr.DateRecorded),
            hr.Disease,
            COALESCE(NULLIF(TRIM(SUBSTRING_INDEX(h.Address, ',', 1)), ''), 'Unknown')
        ORDER BY DATE(hr.DateRecorded), hr.Disease, Barangay
    """

    with get_engine().connect() as connection:
        df = pd.read_sql(text(query), connection)

    if df.empty:
        raise ValueError("No aggregated health record data was returned from the database.")

    df["DateRecorded"] = pd.to_datetime(df["DateRecorded"])
    df["Disease"] = df["Disease"].astype(str).str.strip()
    df["Barangay"] = df["Barangay"].fillna("Unknown").astype(str).str.strip()
    df["Cases"] = df["Cases"].astype(int)

    return df[["DateRecorded", "Disease", "Barangay", "Cases"]]

# =========================================================
# BUILD ML DATASET
# =========================================================
def build_ml_dataset(data):
    if data.empty:
        raise ValueError("Model training requires at least one aggregated case row.")

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

    df = df.dropna().reset_index(drop=True)

    if df.empty:
        raise ValueError(
            "Not enough disease and barangay history is available to build lag features."
        )

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
        ("model", RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    model.fit(X, y)
    return model

# =========================================================
# PREDICTION FUNCTION
# =========================================================
def predict_cases(model, ts, disease_name, barangay_name=None, forecast_days=7):

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
        next_date = pd.to_datetime(last.iloc[-1]["DateRecorded"]) + timedelta(days=1)

        new_data = pd.DataFrame([{
            "Disease": disease_name,
            "Barangay": barangay_name if barangay_name else "Unknown",
            "Lag1": lag1,
            "Lag2": lag2,
            "Lag3": lag3,
            "Month": next_date.month,
            "Day": next_date.day,
            "DayOfWeek": next_date.dayofweek
        }])

        pred = model.predict(new_data)[0]
        pred = float(max(0, round(pred, 2)))

        forecast_values.append(pred)

        working = pd.concat([
            working,
            pd.DataFrame([{
                "DateRecorded": next_date,
                "Cases": pred
            }])
        ])

    return forecast_values

# =========================================================
# API ENDPOINT
# =========================================================
@app.get("/predict")
def predict(request: Request, disease: str, barangay: str = None):
    model = request.app.state.ml_model
    if model is None:
        detail = request.app.state.model_error or "Model is not ready."
        raise HTTPException(status_code=503, detail=detail)

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

    forecast = predict_cases(model, ts, disease, barangay)
    if isinstance(forecast, dict) and "error" in forecast:
        raise HTTPException(status_code=400, detail=forecast["error"])

    return {
        "disease": disease,
        "barangay": barangay,
        "forecast": forecast
    }
