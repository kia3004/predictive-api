# ============================================
# PHIMAS FASTAPI PREDICTIVE MODEL (ARIMA)
# ============================================

from fastapi import FastAPI
import pandas as pd
import numpy as np
import mysql.connector
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error

app = FastAPI()

# ============================================
# DATABASE CONNECTION FUNCTION
# ============================================

def get_data():
    conn = mysql.connector.connect(
        host="centerbeam.proxy.rlwy.net",
        port=38661,
        user="phimas_user",
        password="Phimas123!",
        database="railway"
    )

    query = """
    SELECT DateRecorded, Disease
    FROM healthrecords
    """

    df = pd.read_sql(query, conn)
    conn.close()

    df['DateRecorded'] = pd.to_datetime(df['DateRecorded'])

    data = df.groupby(['DateRecorded', 'Disease']).size().reset_index(name='Cases')

    return data


# ============================================
# CORE PREDICTION FUNCTION
# ============================================

def predict_disease(data, disease_name, forecast_days=7):

    disease_data = data[data['Disease'] == disease_name]

    if disease_data.empty:
        return {"error": "No data found for this disease"}

    ts = disease_data.set_index('DateRecorded')['Cases'].asfreq('D').fillna(0)

    if len(ts) < 10:
        return {"error": "Not enough data for prediction"}

    # TRAIN/TEST SPLIT
    train_size = int(len(ts) * 0.8)
    train, test = ts[:train_size], ts[train_size:]

    # TRAIN MODEL
    model = ARIMA(train, order=(2,1,2))
    model_fit = model.fit()

    # TEST PREDICTION
    pred_test = model_fit.forecast(steps=len(test))

    mae = mean_absolute_error(test, pred_test)
    rmse = np.sqrt(((test - pred_test) ** 2).mean())

    # FINAL MODEL
    final_model = ARIMA(ts, order=(2,1,2)).fit()
    forecast = final_model.forecast(steps=forecast_days)

    # RISK SCORE
    avg_cases = ts.mean()

    def risk_score(x):
        if x > avg_cases * 1.5:
            return "High"
        elif x > avg_cases:
            return "Medium"
        else:
            return "Low"

    risks = [risk_score(x) for x in forecast]

    # CONFIDENCE
    confidence = max(0, 100 - (rmse * 5))

    # TREND
    trend = "Increasing" if forecast.iloc[-1] > forecast.iloc[0] else "Decreasing"

    return {
        "disease": disease_name,
        "forecast": forecast.tolist(),
        "risk_levels": risks,
        "trend": trend,
        "confidence_score": round(confidence, 2),
        "mae": round(mae, 2),
        "rmse": round(rmse, 2)
    }


# ============================================
# API ENDPOINTS
# ============================================

@app.get("/")
def root():
    return {"message": "PHIMAS Predictive API is running"}

@app.get("/predict/{disease}")
def predict(disease: str):
    data = get_data()
    result = predict_disease(data, disease)
    return result


# GET ALL DISEASES
@app.get("/predict_all")
def predict_all():
    data = get_data()
    diseases = data['Disease'].unique()

    results = []

    for d in diseases:
        res = predict_disease(data, d)
        if "error" not in res:
            results.append(res)

    return results