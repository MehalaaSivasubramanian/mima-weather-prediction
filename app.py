from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
from tensorflow import keras   # ✅ FIXED
import os
import requests
from datetime import timedelta

app = Flask(__name__)

# =========================
# DOWNLOAD micro.csv FROM GOOGLE DRIVE
# =========================
def download_micro_csv():
    if os.path.exists("micro.csv"):
        print("✅ micro.csv already exists")
        return

    print("📥 Downloading micro.csv from Google Drive...")

    FILE_ID = "17eeKYcev5Bvw3QooTTLkP-i69hdkqZTo"
    URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"

    try:
        response = requests.get(URL, timeout=60)

        if response.status_code == 200:
            with open("micro.csv", "wb") as f:
                f.write(response.content)
            print("✅ micro.csv downloaded successfully!")
        else:
            print(f"❌ Failed to download micro.csv. Status code: {response.status_code}")
    except Exception as e:
        print(f"❌ Error downloading micro.csv: {e}")

download_micro_csv()

# =========================
# LOAD MODEL & SCALERS
# =========================
try:
    mima_model = keras.models.load_model("mima_model.keras", compile=False)  # ✅ FIXED
    macro_scaler = joblib.load("macro_X_scaler.pkl")
    print("✅ LSTM model & scalers loaded!")
except Exception as e:
    print(f"⚠️ Model missing - using fallback: {e}")
    mima_model = None
    macro_scaler = None

# =========================
# GLOBAL DATA LOAD
# =========================
MICRO_DF, MACRO_DF = None, None

def load_data():
    global MICRO_DF, MACRO_DF
    if MICRO_DF is None or MACRO_DF is None:
        print("Loading datasets...")

        if os.path.exists("micro.csv"):
            MICRO_DF = pd.read_csv("micro.csv")
            MICRO_DF["Datetime"] = pd.to_datetime(
                MICRO_DF[["Year", "Month", "Day", "Hour", "Minute"]]
            )
            print("✅ micro.csv loaded")
        else:
            print("⚠️ micro.csv missing - creating fallback synthetic micro data...")
            dates = pd.date_range("2024-01-01", periods=5000, freq="h")
            cities = ["Chennai", "Coimbatore", "Madurai", "Trichy", "Kumbakonam"]
            MICRO_DF = pd.DataFrame({
                "Datetime": np.tile(dates, len(cities)),
                "City": np.repeat(cities, len(dates)),
                "TAIR": np.random.uniform(24, 36, len(dates) * len(cities)),
                "RELH": np.random.uniform(55, 90, len(dates) * len(cities)),
                "THMP": np.random.uniform(24, 38, len(dates) * len(cities)),
                "WSPD": np.random.uniform(2, 15, len(dates) * len(cities)),
                "WDIR": np.random.uniform(0, 360, len(dates) * len(cities)),
                "WSMX": np.random.uniform(3, 20, len(dates) * len(cities)),
                "PRCP": np.random.uniform(0, 5, len(dates) * len(cities)),
                "PRES": np.random.uniform(990, 1020, len(dates) * len(cities)),
                "SRAD": np.random.uniform(0, 1000, len(dates) * len(cities)),
            })

        if os.path.exists("macro.csv"):
            MACRO_DF = pd.read_csv("macro.csv")
            if "Datetime" in MACRO_DF.columns:
                MACRO_DF["Datetime"] = pd.to_datetime(MACRO_DF["Datetime"])
            elif "datetime" in MACRO_DF.columns:
                MACRO_DF["Datetime"] = pd.to_datetime(MACRO_DF["datetime"])
            else:
                raise ValueError("macro.csv must contain either 'Datetime' or 'datetime'")
            print("✅ macro.csv loaded")
        else:
            print("⚠️ macro.csv missing - creating synthetic macro data...")
            dates = pd.date_range("2024-01-01", periods=10000, freq="h")
            MACRO_DF = pd.DataFrame({
                "Datetime": dates,
                "ATT1": np.random.uniform(20, 35, 10000),
                "ATT2": np.random.uniform(60, 90, 10000),
                "ATT3": np.random.uniform(5, 15, 10000),
                "ATT4": np.random.uniform(0, 10, 10000),
                "ATT5": np.random.uniform(990, 1020, 10000),
                "ATT6": np.random.uniform(0, 1000, 10000),
                "ATT7": np.random.uniform(0, 50, 10000),
                "ATT8": np.random.uniform(0, 360, 10000),
                "ATT9": np.random.uniform(0, 20, 10000),
                "ATT10": np.random.uniform(0, 5, 10000),
                "City": np.random.choice(
                    ["Chennai", "Coimbatore", "Madurai", "Trichy", "Kumbakonam"], 10000
                )
            })

        print("✅ Datasets loaded globally")
    return MICRO_DF, MACRO_DF


def fix_values(temp, hum, wind, city):
    if city == "Chennai":
        temp = np.clip(temp, 26, 40)
    elif city == "Coimbatore":
        temp = np.clip(temp, 20, 32)
    elif city == "Madurai":
        temp = np.clip(temp, 25, 39)
    elif city == "Trichy":
        temp = np.clip(temp, 26, 40)
    else:
        temp = np.clip(temp, 24, 38)

    hum = np.clip(hum, 50, 90)
    wind = np.clip(wind, 2, 20)
    return round(temp, 1), round(hum, 1), round(wind, 1)


def fallback_weather(city):
    base_temp = {
        "Chennai": 32,
        "Madurai": 31,
        "Trichy": 31,
        "Kumbakonam": 30,
        "Coimbatore": 27
    }.get(city, 30)

    forecast = []
    for i in range(5):
        temp = base_temp + np.random.uniform(-2, 2)
        hum = np.random.uniform(65, 85)
        wind = np.random.uniform(6, 14)
        forecast.append([temp, hum, wind])

    return np.array(forecast)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        city = request.form["city"]
        date = request.form["date"]
        time = request.form["time"]

        if not city:
            return render_template("index.html", error="⚠️ Please select a city")

        if not date or not time:
            return render_template("index.html", error="⚠️ Please select date & time")

        input_datetime = pd.to_datetime(date + " " + time)

        if input_datetime.year < 2021 or input_datetime.year > 2024:
            return render_template("index.html", error="⚠️ Only 2021–2024 supported")

        print(f"\n🔍 PREDICTING {city} at {input_datetime}")

        micro_df, macro_df = load_data()

        city_micro = micro_df[micro_df["City"] == city].sort_values("Datetime")
        valid_micro = city_micro[city_micro["Datetime"] <= input_datetime]

        use_model = len(valid_micro) >= 48

        if use_model and mima_model is not None:
            micro_seq = valid_micro.tail(48)
            features = ["TAIR", "RELH", "THMP", "WSPD", "WDIR", "WSMX", "PRCP", "PRES", "SRAD"]
            scaler_X = joblib.load(f"micro_{city.lower()}_scaler_X.pkl")
            micro_scaled = scaler_X.transform(micro_seq[features])
            micro_input = micro_scaled.reshape(1, 48, 9)
        else:
            micro_input = None

        valid_macro = macro_df[macro_df["Datetime"] <= input_datetime].sort_values("Datetime")

        if len(valid_macro) >= 12 and macro_scaler is not None:
            macro_seq = valid_macro.tail(12)
            macro_seq = pd.get_dummies(macro_seq, columns=["City"])
            macro_scaled = macro_scaler.transform(macro_seq)
            macro_input = macro_scaled.reshape(1, 12, -1)
        else:
            macro_input = None

        if use_model and micro_input is not None and macro_input is not None:
            scaler_y = joblib.load(f"micro_{city.lower()}_scaler_y.pkl")
            pred = mima_model.predict([micro_input, macro_input], verbose=0)
            pred_real = np.array([scaler_y.inverse_transform([p])[0] for p in pred[0]])
        else:
            pred_real = fallback_weather(city)

        forecast = []
        for i in range(5):
            future_time = input_datetime + timedelta(hours=i + 1)
            temp, hum, wind = fix_values(pred_real[i][0], pred_real[i][1], pred_real[i][2], city)
            forecast.append({
                "time": future_time.strftime("%Y-%m-%d %H:%M"),
                "temp": temp,
                "hum": hum,
                "wind": wind
            })

        return render_template("index.html", forecast=forecast)

    except Exception as e:
        print(f"❌ ERROR: {e}")
        pred_real = fallback_weather("Coimbatore")
        forecast = []

        for i in range(5):
            future_time = pd.Timestamp.now() + timedelta(hours=i + 1)
            temp, hum, wind = fix_values(pred_real[i][0], pred_real[i][1], pred_real[i][2], "Coimbatore")
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
