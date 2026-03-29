from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
import keras
import os
import requests
from datetime import timedelta

app = Flask(__name__)

def download_micro_csv():
    if os.path.exists("micro.csv"):
        print("✅ micro.csv exists")
        return
    print("📥 Downloading micro.csv...")
    FILE_ID = "17eeKYcev5Bvw3QooTTLkP-i69hdkqZTo"
    URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
    try:
        response = requests.get(URL, timeout=60)
        if response.status_code == 200:
            with open("micro.csv", "wb") as f:
                f.write(response.content)
            print("✅ micro.csv downloaded")
    except Exception as e:
        print(f"❌ micro.csv download failed: {e}")

download_micro_csv()

# Load models (strict - fail if missing)
mima_model = None
macro_scaler = None
try:
    mima_model = keras.models.load_model("mima_model.keras")
    macro_scaler = joblib.load("macro_X_scaler.pkl")
    print("✅ Models loaded")
except:
    print("⚠️ Missing model files")

MICRO_DF, MACRO_DF = None, None

def load_data():
    global MICRO_DF, MACRO_DF
    
    # MICRO - STRICT REQUIREMENT
    if not os.path.exists("micro.csv"):
        raise FileNotFoundError("micro.csv required")
    
    MICRO_DF = pd.read_csv("micro.csv")
    MICRO_DF["Datetime"] = pd.to_datetime(MICRO_DF[["Year", "Month", "Day", "Hour", "Minute"]])
    
    # MACRO - STRICT REQUIREMENT  
    if not os.path.exists("macro.csv"):
        raise FileNotFoundError("macro.csv required")
    
    MACRO_DF = pd.read_csv("macro.csv")
    if "datetime" in MACRO_DF.columns:
        MACRO_DF["Datetime"] = pd.to_datetime(MACRO_DF["datetime"])
    elif "Datetime" in MACRO_DF.columns:
        MACRO_DF["Datetime"] = pd.to_datetime(MACRO_DF["Datetime"])
    else:
        raise ValueError("macro.csv missing datetime column")
    
    print(f"✅ Loaded: micro={len(MICRO_DF)} rows, macro={len(MACRO_DF)} rows")
    return MICRO_DF, MACRO_DF

def fix_values(temp, hum, wind, city):
    if city == "Chennai": temp = np.clip(temp, 26, 40)
    elif city == "Coimbatore": temp = np.clip(temp, 20, 32)
    elif city == "Madurai": temp = np.clip(temp, 25, 39)
    elif city == "Trichy": temp = np.clip(temp, 26, 40)
    else: temp = np.clip(temp, 24, 38)
    
    hum = np.clip(hum, 50, 90)
    wind = np.clip(wind, 2, 20)
    return round(temp, 1), round(hum, 1), round(wind, 1)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    city = request.form.get("city")
    date = request.form.get("date")
    time = request.form.get("time")

    if not all([city, date, time]):
        return render_template("index.html", warning="⚠️ All fields required")

    input_datetime = pd.to_datetime(f"{date} {time}")
    
    if not (pd.Timestamp("2021-01-01") <= input_datetime <= pd.Timestamp("2024-12-31")):
        return render_template("index.html", warning="⚠️ Date must be 2021-2024")

    # STRICT DATA & MODEL CHECKS
    if mima_model is None or macro_scaler is None:
        return render_template("index.html", warning="⚠️ Model files missing")

    try:
        micro_df, macro_df = load_data()
    except Exception as e:
        return render_template("index.html", warning=f"⚠️ Data error: {str(e)}")

    # STRICT MICRO DATA CHECK
    city_micro = micro_df[micro_df["City"] == city].sort_values("Datetime")
    valid_micro = city_micro[city_micro["Datetime"] <= input_datetime]
    
    if len(valid_micro) < 48:
        return render_template("index.html", warning=f"⚠️ Need 48+ {city} micro rows before {input_datetime}")

    # STRICT MACRO DATA CHECK  
    valid_macro = macro_df[macro_df["Datetime"] <= input_datetime]
    if len(valid_macro) < 12:
        return render_template("index.html", warning=f"⚠️ Need 12+ macro rows before {input_datetime}")

    # STRICT SCALER CHECK
    scaler_X_path = f"micro_{city.lower()}_scaler_X.pkl"
    scaler_y_path = f"micro_{city.lower()}_scaler_y.pkl"
    if not all(os.path.exists(p) for p in [scaler_X_path, scaler_y_path]):
        return render_template("index.html", warning=f"⚠️ Missing {city} scalers")

    # PROCESS MICRO (REAL DATA ONLY)
    micro_seq = valid_micro.tail(48)
    features = ["TAIR", "RELH", "THMP", "WSPD", "WDIR", "WSMX", "PRCP", "PRES", "SRAD"]
    micro_features = micro_seq[features]

    scaler_X = joblib.load(scaler_X_path)
    scaler_y = joblib.load(scaler_y_path)
    
    micro_scaled = scaler_X.transform(micro_features)
    micro_input = micro_scaled.reshape(1, 48, 9)

    # PROCESS MACRO (REAL DATA ONLY)
    macro_seq = valid_macro.tail(12).copy()
    macro_seq = macro_seq[["ATT1", "ATT2", "ATT3", "ATT4", "ATT5",
                          "ATT6", "ATT7", "ATT8", "ATT9", "ATT10", "City"]]
    
    macro_seq = pd.get_dummies(macro_seq, columns=["City"])
    expected_cols = ["ATT1", "ATT2", "ATT3", "ATT4", "ATT5", "ATT6", "ATT7", "ATT8", "ATT9", "ATT10",
                    "City_Chennai", "City_Coimbatore", "City_Kumbakonam", "City_Madurai", "City_Trichy"]
    
    for col in expected_cols:
        if col not in macro_seq: macro_seq[col] = 0
    macro_seq = macro_seq[expected_cols]
    
    macro_scaled = macro_scaler.transform(macro_seq)
    macro_input = macro_scaled.reshape(1, 12, 15)

    # REAL MODEL PREDICTION
    pred = mima_model.predict([micro_input, macro_input], verbose=0)
    
    predictions = []
    for step in range(5):
        pred_real = scaler_y.inverse_transform([pred[0, step, :]])[0]
        predictions.append(pred_real)

    # REAL FORECAST OUTPUT
    forecast = []
    for i, pred_real in enumerate(predictions):
        future_time = input_datetime + timedelta(hours=i + 1)
        temp, hum, wind = fix_values(pred_real[0], pred_real[1], pred_real[2], city)
        forecast.append({
            "time": future_time.strftime("%Y-%m-%d %H:%M"),
            "temp": temp,
            "hum": hum,
            "wind": wind
        })

    return render_template("index.html", forecast=forecast)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
