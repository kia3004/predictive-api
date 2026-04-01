from __future__ import annotations

from contextlib import asynccontextmanager
from functools import lru_cache
import os
from typing import Any

from fastapi import FastAPI, HTTPException, Request
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


WEEK_INDEX_BASE = pd.Timestamp("2000-01-03")
FEATURE_COLUMNS = ["Disease", "Barangay", "Lag1", "Lag2", "Lag3", "Week"]


class ConstantRiskClassifier:
    def __init__(self, constant: int):
        self.constant = int(constant)

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "ConstantRiskClassifier":
        return self

    def predict(self, X: pd.DataFrame) -> list[int]:
        return [self.constant] * len(X)


def get_env(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


def normalize_database_url(database_url: str) -> str:
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


def empty_weekly_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["WeekStart", "Week", "Disease", "Barangay", "Cases"])


def week_start_to_index(value: pd.Series | pd.Timestamp) -> pd.Series | int:
    if isinstance(value, pd.Series):
        return ((pd.to_datetime(value) - WEEK_INDEX_BASE).dt.days // 7).astype(int)
    return int((pd.Timestamp(value) - WEEK_INDEX_BASE).days // 7)


def calculate_risk_level(cases: float) -> int:
    risk_score = max(float(cases), 0.0) * 5.0
    if risk_score <= 30:
        return 0
    if risk_score <= 70:
        return 1
    return 2


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["Disease", "Barangay"]),
            ("num", StandardScaler(), ["Lag1", "Lag2", "Lag3", "Week"]),
        ]
    )


def build_regression_models() -> dict[str, Pipeline]:
    return {
        "LinearRegression": Pipeline(
            steps=[("prep", build_preprocessor()), ("model", LinearRegression())]
        ),
        "RandomForest": Pipeline(
            steps=[
                ("prep", build_preprocessor()),
                ("model", RandomForestRegressor(n_estimators=200, random_state=42)),
            ]
        ),
    }


def build_classification_models(training_rows: int) -> dict[str, Pipeline]:
    neighbors = max(1, min(5, training_rows))
    return {
        "LogisticRegression": Pipeline(
            steps=[
                ("prep", build_preprocessor()),
                ("model", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        ),
        "KNN": Pipeline(
            steps=[
                ("prep", build_preprocessor()),
                ("model", KNeighborsClassifier(n_neighbors=neighbors)),
            ]
        ),
    }


def get_data(barangay_name: str | None = None, disease_name: str | None = None) -> pd.DataFrame:
    filters: list[str] = [
        "hr.DateRecorded IS NOT NULL",
        "hr.PatientID IS NOT NULL",
        "hr.Disease IS NOT NULL",
        "TRIM(hr.Disease) <> ''",
    ]
    params: dict[str, Any] = {}

    if disease_name:
        filters.append("LOWER(TRIM(hr.Disease)) = LOWER(:disease_name)")
        params["disease_name"] = disease_name.strip()

    if barangay_name:
        filters.append(
            """
            LOWER(
                COALESCE(
                    NULLIF(TRIM(SUBSTRING_INDEX(h.Address, ',', 1)), ''),
                    'Unknown'
                )
            ) = LOWER(:barangay_name)
            """
        )
        params["barangay_name"] = barangay_name.strip()

    query = f"""
        SELECT
            DATE_SUB(hr.DateRecorded, INTERVAL WEEKDAY(hr.DateRecorded) DAY) AS WeekStart,
            TRIM(hr.Disease) AS Disease,
            COALESCE(NULLIF(TRIM(SUBSTRING_INDEX(h.Address, ',', 1)), ''), 'Unknown') AS Barangay,
            COUNT(*) AS Cases
        FROM health_records hr
        INNER JOIN household_members hm ON hr.PatientID = hm.PatientID
        INNER JOIN households h ON hm.HouseholdID = h.HouseholdID
        WHERE {" AND ".join(filters)}
        GROUP BY WeekStart, Disease, Barangay
        ORDER BY WeekStart, Disease, Barangay
    """

    with get_engine().connect() as connection:
        df = pd.read_sql(text(query), connection, params=params)

    if df.empty:
        return empty_weekly_frame()

    df["WeekStart"] = pd.to_datetime(df["WeekStart"])
    df["Disease"] = df["Disease"].astype(str).str.strip()
    df["Barangay"] = df["Barangay"].astype(str).str.strip()
    df["Cases"] = df["Cases"].astype(int)

    return complete_weekly_series(df)


def complete_weekly_series(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return empty_weekly_frame()

    completed_groups: list[pd.DataFrame] = []

    for (disease, barangay), group in df.groupby(["Disease", "Barangay"], sort=False):
        ordered = (
            group[["WeekStart", "Cases"]]
            .drop_duplicates(subset=["WeekStart"], keep="last")
            .sort_values("WeekStart")
        )

        full_weeks = pd.date_range(
            start=ordered["WeekStart"].min(),
            end=ordered["WeekStart"].max(),
            freq="W-MON",
        )

        expanded = (
            ordered.set_index("WeekStart")
            .reindex(full_weeks, fill_value=0)
            .rename_axis("WeekStart")
            .reset_index()
        )

        expanded["Disease"] = disease
        expanded["Barangay"] = barangay
        expanded["Cases"] = expanded["Cases"].astype(int)
        completed_groups.append(expanded)

    completed = pd.concat(completed_groups, ignore_index=True)
    completed["Week"] = week_start_to_index(completed["WeekStart"])

    return completed[["WeekStart", "Week", "Disease", "Barangay", "Cases"]].sort_values(
        ["Disease", "Barangay", "Week"]
    ).reset_index(drop=True)


def build_ml_dataset(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        raise ValueError("No weekly PHIMAS health record data was found.")

    df = data.copy().sort_values(["Disease", "Barangay", "Week"]).reset_index(drop=True)
    grouped = df.groupby(["Disease", "Barangay"], sort=False)

    df["Lag1"] = grouped["Cases"].shift(1)
    df["Lag2"] = grouped["Cases"].shift(2)
    df["Lag3"] = grouped["Cases"].shift(3)

    model_df = df.dropna(subset=["Lag1", "Lag2", "Lag3"]).copy()
    if model_df.empty:
        raise ValueError(
            "Model training requires at least 4 weekly observations in a disease/barangay series "
            "(3 lag weeks plus 1 target week)."
        )

    model_df["Lag1"] = model_df["Lag1"].astype(float)
    model_df["Lag2"] = model_df["Lag2"].astype(float)
    model_df["Lag3"] = model_df["Lag3"].astype(float)
    model_df["RiskLevel"] = model_df["Cases"].apply(calculate_risk_level).astype(int)

    return model_df.reset_index(drop=True)


def split_training_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if len(df) < 5:
        return df.copy(), df.copy(), "train_only"

    split_index = int(len(df) * 0.8)
    split_index = max(1, min(split_index, len(df) - 1))
    return df.iloc[:split_index].copy(), df.iloc[split_index:].copy(), "holdout"


def train_model() -> dict[str, Any]:
    weekly_data = get_data()
    if weekly_data.empty:
        raise ValueError("No weekly PHIMAS health record data was found in the database.")

    dataset = build_ml_dataset(weekly_data)
    dataset = dataset.sort_values(["Week", "Disease", "Barangay"]).reset_index(drop=True)

    train_df, test_df, evaluation_mode = split_training_frame(dataset)

    X_train = train_df[FEATURE_COLUMNS]
    X_test = test_df[FEATURE_COLUMNS]
    y_train_reg = train_df["Cases"]
    y_test_reg = test_df["Cases"]
    y_train_class = train_df["RiskLevel"]
    y_test_class = test_df["RiskLevel"]

    best_reg_model = None
    best_reg_name = None
    best_reg_score = float("inf")
    reg_scores: dict[str, float] = {}
    reg_errors: dict[str, str] = {}

    for name, model in build_regression_models().items():
        try:
            model.fit(X_train, y_train_reg)
            predictions = model.predict(X_test)
            mae = float(mean_absolute_error(y_test_reg, predictions))
            reg_scores[name] = mae

            if mae < best_reg_score:
                best_reg_score = mae
                best_reg_model = model
                best_reg_name = name
        except Exception as exc:
            reg_errors[name] = str(exc)

    if best_reg_model is None or best_reg_name is None:
        raise ValueError(f"Regression model training failed: {reg_errors or 'no model could be trained'}")

    best_class_model = None
    best_class_name = None
    best_class_score = float("-inf")
    class_scores: dict[str, float] = {}
    class_errors: dict[str, str] = {}

    for name, model in build_classification_models(len(X_train)).items():
        try:
            model.fit(X_train, y_train_class)
            predictions = model.predict(X_test)
            accuracy = float(accuracy_score(y_test_class, predictions))
            class_scores[name] = accuracy

            if accuracy > best_class_score:
                best_class_score = accuracy
                best_class_model = model
                best_class_name = name
        except Exception as exc:
            class_errors[name] = str(exc)

    if best_class_model is None or best_class_name is None:
        fallback_class = int(y_train_class.mode().iloc[0])
        best_class_model = ConstantRiskClassifier(fallback_class).fit(X_train, y_train_class)
        best_class_name = "ConstantRiskClassifier"
        fallback_predictions = best_class_model.predict(X_test)
        best_class_score = float(accuracy_score(y_test_class, fallback_predictions))
        class_scores[best_class_name] = best_class_score

    return {
        "best_reg": best_reg_model,
        "best_reg_name": best_reg_name,
        "best_reg_mae": best_reg_score,
        "best_class": best_class_model,
        "best_class_name": best_class_name,
        "best_class_accuracy": best_class_score,
        "regression_scores": reg_scores,
        "classification_scores": class_scores,
        "regression_errors": reg_errors,
        "classification_errors": class_errors,
        "training_rows": len(dataset),
        "latest_week": int(weekly_data["Week"].max()),
        "evaluation_mode": evaluation_mode,
    }


def ensure_model_ready(app_instance: FastAPI) -> dict[str, Any]:
    if getattr(app_instance.state, "ml_model", None) is not None:
        return app_instance.state.ml_model

    try:
        app_instance.state.ml_model = train_model()
        app_instance.state.model_error = None
        return app_instance.state.ml_model
    except Exception as exc:
        app_instance.state.model_error = str(exc)
        raise HTTPException(status_code=503, detail=f"Prediction model unavailable: {exc}") from exc


def build_prediction_input(
    weekly_data: pd.DataFrame,
    barangay: str,
    disease: str,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    if weekly_data.empty:
        raise ValueError("No weekly data is available for prediction.")

    normalized_barangay = barangay.strip().lower()
    normalized_disease = disease.strip().lower()

    filtered = weekly_data[
        (weekly_data["Barangay"].str.lower() == normalized_barangay)
        & (weekly_data["Disease"].str.lower() == normalized_disease)
    ].sort_values("Week")

    if filtered.empty:
        raise LookupError(
            f"No weekly data found for disease '{disease}' in barangay '{barangay}'."
        )

    if len(filtered) < 3:
        raise ValueError(
            f"At least 3 weeks of weekly data are required to predict '{disease}' for '{barangay}'."
        )

    last_three = filtered.tail(3).reset_index(drop=True)
    next_week_start = pd.Timestamp(last_three.iloc[-1]["WeekStart"]) + pd.Timedelta(days=7)
    next_week_index = week_start_to_index(next_week_start)

    features = pd.DataFrame(
        [
            {
                "Disease": str(last_three.iloc[-1]["Disease"]),
                "Barangay": str(last_three.iloc[-1]["Barangay"]),
                "Lag1": float(last_three.iloc[-1]["Cases"]),
                "Lag2": float(last_three.iloc[-2]["Cases"]),
                "Lag3": float(last_three.iloc[-3]["Cases"]),
                "Week": int(next_week_index),
            }
        ]
    )

    return features, next_week_start


def predict_barangay_disease(
    weekly_data: pd.DataFrame,
    barangay: str,
    disease: str,
    model_bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    models = model_bundle or train_model()
    X_input, next_week_start = build_prediction_input(weekly_data, barangay, disease)

    predicted_cases = max(0.0, float(models["best_reg"].predict(X_input)[0]))
    risk_level = int(models["best_class"].predict(X_input)[0])

    return {
        "barangay": barangay.strip(),
        "disease": disease.strip(),
        "predicted_cases": round(predicted_cases, 2),
        "risk_level": risk_level,
        "forecast_week_start": next_week_start.date().isoformat(),
        "confidence_ratio": round(float(models.get("best_class_accuracy", 0.0)), 4),
    }


def save_prediction_to_db(result: dict[str, Any]) -> dict[str, Any]:
    if "barangay" not in result or "disease" not in result or "predicted_cases" not in result:
        raise ValueError("Prediction result must include barangay, disease, and predicted_cases.")

    confidence = float(result.get("confidence_ratio", 0.0))
    query = text(
        """
        INSERT INTO predictive_analysis
            (DateGenerated, Disease, PredictedCases, HighRiskBarangay, ConfidenceScore)
        VALUES
            (CURRENT_DATE(), :disease, :predicted_cases, :barangay, :confidence)
        """
    )

    with get_engine().begin() as connection:
        connection.execute(
            query,
            {
                "disease": str(result["disease"]).strip(),
                "predicted_cases": int(round(float(result["predicted_cases"]))),
                "barangay": str(result["barangay"]).strip(),
                "confidence": round(confidence, 2),
            },
        )

    return {"status": "saved"}


def compute_high_risk_barangays(
    weekly_data: pd.DataFrame,
    model_bundle: dict[str, Any],
) -> list[dict[str, Any]]:
    if weekly_data.empty:
        return []

    results: list[dict[str, Any]] = []

    for (disease, barangay), group in weekly_data.groupby(["Disease", "Barangay"], sort=False):
        if len(group) < 3:
            continue

        try:
            prediction = predict_barangay_disease(
                weekly_data=weekly_data,
                barangay=barangay,
                disease=disease,
                model_bundle=model_bundle,
            )
        except (LookupError, ValueError):
            continue

        if prediction["risk_level"] == 2:
            results.append(
                {
                    "barangay": prediction["barangay"],
                    "disease": prediction["disease"],
                    "predicted_cases": prediction["predicted_cases"],
                    "risk_level": prediction["risk_level"],
                }
            )

    return sorted(results, key=lambda item: item["predicted_cases"], reverse=True)


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    app_instance.state.ml_model = None
    app_instance.state.model_error = None

    try:
        app_instance.state.ml_model = train_model()
        app_instance.state.model_error = None
        print("Prediction model trained successfully.")
    except Exception as exc:
        app_instance.state.ml_model = None
        app_instance.state.model_error = str(exc)
        print(f"Prediction model training failed: {exc}")

    yield

    get_engine().dispose()


app = FastAPI(lifespan=lifespan)


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "PHIMAS prediction API", "status": "running"}


@app.get("/health")
def health(request: Request) -> dict[str, Any]:
    model_bundle = getattr(request.app.state, "ml_model", None)
    model_error = getattr(request.app.state, "model_error", None)

    if model_bundle is None:
        return {
            "status": "degraded",
            "model_ready": False,
            "detail": model_error or "Prediction model is not trained yet.",
        }

    return {
        "status": "ok",
        "model_ready": True,
        "best_regression_model": model_bundle["best_reg_name"],
        "best_classification_model": model_bundle["best_class_name"],
        "regression_mae": round(float(model_bundle["best_reg_mae"]), 4),
        "classification_accuracy": round(float(model_bundle["best_class_accuracy"]), 4),
        "training_rows": int(model_bundle["training_rows"]),
        "evaluation_mode": model_bundle["evaluation_mode"],
    }


@app.get("/predict")
def predict(request: Request, disease: str, barangay: str) -> dict[str, Any]:
    if not disease.strip() or not barangay.strip():
        raise HTTPException(status_code=422, detail="Both disease and barangay are required.")

    model_bundle = ensure_model_ready(request.app)
    weekly_data = get_data()
    if weekly_data.empty:
        raise HTTPException(status_code=503, detail="No weekly PHIMAS health record data was found.")

    try:
        prediction = predict_barangay_disease(
            weekly_data=weekly_data,
            barangay=barangay,
            disease=disease,
            model_bundle=model_bundle,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    return {
        "barangay": prediction["barangay"],
        "disease": prediction["disease"],
        "predicted_cases": prediction["predicted_cases"],
        "risk_level": prediction["risk_level"],
    }


@app.get("/high-risk-barangays")
def high_risk_barangays(request: Request) -> list[dict[str, Any]]:
    model_bundle = ensure_model_ready(request.app)
    weekly_data = get_data()
    if weekly_data.empty:
        raise HTTPException(status_code=503, detail="No weekly PHIMAS health record data was found.")

    try:
        return compute_high_risk_barangays(weekly_data, model_bundle)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"High-risk barangay prediction failed: {exc}") from exc
