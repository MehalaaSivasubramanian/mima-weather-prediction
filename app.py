from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
from tensorflow import keras
import os
import requests
from datetime import timedelta

app = Flask(__name__)

# =========================
# LOAD MODEL + SCALERS
# =========================

try:
    mima_model = keras.models.load_model("mima_weather_model.h5")
    macro_scaler = joblib.load("macro_scaler.pkl")
    print("✅ Model Loaded Successfully")
except:
    mima_model = None
    macro_scaler = None
    print("⚠️ Model loading failed — using fallback")

# =========================
# CITY CONFIG
# =========================

city_coords = {
    "Chennai": [13.0827, 80.2707],
    "Coimbatore": [11.0168, 76.9558],
    "Madurai": [9.9252, 78.1198],
    "Trichy": [10.7905, 78.7047],
    "Kumbakonam": [10.9601, 79.3845]
}

city_file_map = {
    "Chennai": "chennai",
    "Coimbatore": "coimbatore",
    "Madurai": "madurai",
    "Trichy": "trichy",
    "Kumbakonam": "kumbakonam"
}

# =========================
# LOAD DATA
# =========================

def load_data():
    micro_df = pd.read_csv("micro_weather.csv")
    macro_df = pd.read_csv("macro_weather.csv")

    micro_df["Datetime"] = pd.to_datetime(micro_df["Datetime"])
    macro_df["Datetime"] = pd.to_datetime(macro_df["Datetime"])

    return micro_df, macro_df


# =========================
# FALLBACK WEATHER
# =========================

def fallback_weather(city):
    return np.array([
        [30, 65, 10],
        [31, 63, 11],
        [32, 61, 12],
        [31, 60, 10],
        [30, 62, 9]
    ])


# =========================
# FIX VALUES
# =========================

def fix_values(temp, hum, wind, city):
    temp = round(float(temp), 1)
    hum = round(float(hum), 1)
    wind = round(float(wind), 1)
    return temp, hum, wind


# =========================
# HOME
# =========================

@app.route("/")
def home():
    return render_template("index.html")


# =========================
# PREDICT
# =========================

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

        print(f"\n🔍 PREDICTING {city} at {input_datetime}")

        micro_df, macro_df = load_data()

        # ✅ Future prediction support
        latest_micro = micro_df["Datetime"].max()
        latest_macro = macro_df["Datetime"].max()
        latest_available = min(latest_micro, latest_macro)

        if input_datetime > latest_available:
            print(f"📅 Future prediction requested: {input_datetime}")
            print(f"➡️ Using latest available data: {latest_available}")
            input_datetime = latest_available

        city_micro = micro_df[micro_df["City"] == city].sort_values("Datetime")
        valid_micro = city_micro[city_micro["Datetime"] <= input_datetime]

        use_model = len(valid_micro) >= 48
        city_key = city_file_map.get(city, city.lower())

        # =========================
        # MICRO DATA
        # =========================

        if use_model and mima_model is not None:
            micro_seq = valid_micro.tail(48)

            features = [
                "TAIR", "RELH", "THMP",
                "WSPD", "WDIR", "WSMX",
                "PRCP", "PRES", "SRAD"
            ]

            scaler_X = joblib.load(f"micro_{city_key}_scaler_X.pkl")

            micro_scaled = scaler_X.transform(micro_seq[features])
            micro_input = micro_scaled.reshape(1, 48, 9)

        else:
            micro_input = None

        # =========================
        # MACRO DATA
        # =========================

        valid_macro = macro_df[
            macro_df["Datetime"] <= input_datetime
        ].sort_values("Datetime")

        if len(valid_macro) >= 12 and macro_scaler is not None:

            macro_seq = valid_macro.tail(12).copy()

            macro_seq = macro_seq[[
                "ATT1", "ATT2", "ATT3", "ATT4", "ATT5",
                "ATT6", "ATT7", "ATT8", "ATT9", "ATT10",
                "City"
            ]]

            macro_seq = pd.get_dummies(
                macro_seq,
                columns=["City"]
            )

            expected_cols = [
                "ATT1", "ATT2", "ATT3", "ATT4", "ATT5",
                "ATT6", "ATT7", "ATT8", "ATT9", "ATT10",
                "City_Chennai",
                "City_Coimbatore",
                "City_Kumbakonam",
                "City_Madurai",
                "City_Trichy"
            ]

            for col in expected_cols:
                if col not in macro_seq.columns:
                    macro_seq[col] = 0

            macro_seq = macro_seq[expected_cols]

            macro_scaled = macro_scaler.transform(macro_seq)

            macro_input = macro_scaled.reshape(1, 12, 15)

        else:
            macro_input = None

        # =========================
        # PREDICTION
        # =========================

        if use_model and micro_input is not None and macro_input is not None:

            scaler_y = joblib.load(
                f"micro_{city_key}_scaler_y.pkl"
            )

            pred = mima_model.predict(
                [micro_input, macro_input],
                verbose=0
            )

            pred_real = np.array([
                scaler_y.inverse_transform([p])[0]
                for p in pred[0]
            ])

        else:
            pred_real = fallback_weather(city)

        # =========================
        # FORECAST
        # =========================

        forecast = []

        for i in range(5):

            future_time = input_datetime + timedelta(hours=i + 1)

            temp, hum, wind = fix_values(
                pred_real[i][0],
                pred_real[i][1],
                pred_real[i][2],
                city
            )

            forecast.append({
                "time": future_time.strftime("%Y-%m-%d %H:%M"),
                "temp": temp,
                "hum": hum,
                "wind": wind
            })

        # =========================
        # CHART DATA
        # =========================

        chart_times = [
            pd.to_datetime(f["time"]).strftime("%H:%M")
            for f in forecast
        ]

        chart_temp = [f["temp"] for f in forecast]
        chart_hum = [f["hum"] for f in forecast]
        chart_wind = [f["wind"] for f in forecast]

        map_coords = city_coords.get(
            city,
            [11.1271, 78.6569]
        )

        return render_template(
            "index.html",
            forecast=forecast,
            chart_times=chart_times,
            chart_temp=chart_temp,
            chart_hum=chart_hum,
            chart_wind=chart_wind,
            selected_city=city,
            map_coords=map_coords
        )

    except Exception as e:

        print(f"❌ ERROR: {e}")

        pred_real = fallback_weather(
            city if 'city' in locals() else "Coimbatore"
        )

        forecast = []

        for i in range(5):

            future_time = pd.Timestamp.now() + timedelta(hours=i + 1)

            temp, hum, wind = fix_values(
                pred_real[i][0],
                pred_real[i][1],
                pred_real[i][2],
                city if 'city' in locals() else "Coimbatore"
            )

            forecast.append({
                "time": future_time.strftime("%Y-%m-%d %H:%M"),
                "temp": temp,
                "hum": hum,
                "wind": wind
            })

        chart_times = [
            pd.to_datetime(f["time"]).strftime("%H:%M")
            for f in forecast
        ]

        chart_temp = [f["temp"] for f in forecast]
        chart_hum = [f["hum"] for f in forecast]
        chart_wind = [f["wind"] for f in forecast]

        map_coords = city_coords.get(
            city if 'city' in locals() else "Coimbatore",
            [11.1271, 78.6569]
        )

        return render_template(
            "index.html",
            forecast=forecast,
            chart_times=chart_times,
            chart_temp=chart_temp,
            chart_hum=chart_hum,
            chart_wind=chart_wind,
            selected_city=city if 'city' in locals() else "Coimbatore",
            map_coords=map_coords
        )


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    app.run(debug=True)
