from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
import keras
import os
import requests
from datetime import timedelta

app = Flask(__name__)

# =========================
# DOWNLOAD micro.csv FROM GOOGLE DRIVE
# =========================
def download_micro_csv():
    """Download micro.csv from Google Drive if not already present"""
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

# Download micro.csv on startup
download_micro_csv()

# =========================
# LOAD MODEL & SCALERS
# =========================
try:
    mima_model = keras.models.load_model("mima_model.keras")
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

        # ---- MICRO DATA ----
        if os.path.exists("micro.csv"):
            MICRO_DF = pd.read_csv("micro.csv")
            MICRO_DF["Datetime"] = pd.to_datetime(
                MICRO_DF[["Year", "Month", "Day", "Hour", "Minute"]]
            )
            print("✅ micro.csv loaded")
        else:
            raise FileNotFoundError("micro.csv not found")

        # ---- MACRO DATA ----
        if os.path.exists("macro.csv"):
            MACRO_DF = pd.read_csv("macro.csv")
            if "Datetime" in MACRO_DF.columns:
                MACRO_DF["Datetime"] = pd.to_datetime(MACRO_DF["Datetime"])
            elif "datetime" in MACRO_DF.columns:
                MACRO_DF["Datetime"] = pd.to_datetime(MACRO_DF["datetime"])
            else:
                raise ValueError("macro.csv must contain either 'Datetime' or 'datetime' column")
            print("✅ macro.csv loaded")
        else:
            raise FileNotFoundError("macro.csv not found")

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


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    city = request.form.get("city")
    date = request.form.get("date")
    time = request.form.get("time")

    # =========================
    # INPUT CHECK
    # =========================
    if not date or not time:
        return render_template(
            "index.html",
            warning="⚠️ Please select both date and time"
        )

    # =========================
    # SAFE DATE PARSE
    # =========================
    try:
        input_datetime = pd.to_datetime(f"{date} {time}")
    except Exception:
        return render_template(
            "index.html",
            warning="⚠️ Please enter a valid date and time"
        )

    print(f"\n🔍 PREDICTING {city} at {input_datetime}")

    # =========================
    # DATE RANGE VALIDATION
    # =========================
    min_date = pd.Timestamp("2021-01-01 00:00")
    max_date = pd.Timestamp("2024-12-31 23:59")

    if input_datetime < min_date or input_datetime > max_date:
        print("⚠️ Unsupported date selected")
        return render_template(
            "index.html",
            warning="⚠️ Only 2021–2024 supported"
        )

    try:
        micro_df, macro_df = load_data()

        # =========================
        # MICRO DATA
        # =========================
        city_micro = micro_df[micro_df["City"] == city].sort_values("Datetime")
        valid_micro = city_micro[city_micro["Datetime"] <= input_datetime]

        print(f"📊 Micro: {len(city_micro)} total, {len(valid_micro)} <= input")

        if len(valid_micro) < 48:
            return render_template(
                "index.html",
                warning="⚠️ Not enough historical micro data for this date/time"
            )

        if mima_model is None:
            return render_template(
                "index.html",
                warning="⚠️ Model file not loaded properly"
            )

        micro_seq = valid_micro.tail(48)
        features = ["TAIR", "RELH", "THMP", "WSPD", "WDIR", "WSMX", "PRCP", "PRES", "SRAD"]
        micro_features = micro_seq[features]

        scaler_X_path = f"micro_{city.lower()}_scaler_X.pkl"
        scaler_y_path = f"micro_{city.lower()}_scaler_y.pkl"

        if not os.path.exists(scaler_X_path) or not os.path.exists(scaler_y_path):
            return render_template(
                "index.html",
                warning=f"⚠️ Missing scaler files for {city}"
            )

        scaler_X = joblib.load(scaler_X_path)
        scaler_y = joblib.load(scaler_y_path)

        micro_scaled = scaler_X.transform(micro_features)
        micro_input = micro_scaled.reshape(1, 48, 9)
        print(f"✅ Micro input ready: {micro_input.shape}")

        # =========================
        # MACRO DATA
        # =========================
        valid_macro = macro_df[macro_df["Datetime"] <= input_datetime].sort_values("Datetime")
        print(f"📊 Macro: {len(valid_macro)} rows <= input")

        if len(valid_macro) < 12:
            return render_template(
                "index.html",
                warning="⚠️ Not enough historical macro data for this date/time"
            )

        if macro_scaler is None:
            return render_template(
                "index.html",
                warning="⚠️ Macro scaler not loaded properly"
            )

        macro_seq = valid_macro.tail(12).copy()
        macro_seq = macro_seq[[
            "ATT1", "ATT2", "ATT3", "ATT4", "ATT5",
            "ATT6", "ATT7", "ATT8", "ATT9", "ATT10", "City"
        ]]
        macro_seq = pd.get_dummies(macro_seq, columns=["City"])

        expected_cols = [
            "ATT1", "ATT2", "ATT3", "ATT4", "ATT5",
            "ATT6", "ATT7", "ATT8", "ATT9", "ATT10",
            "City_Chennai", "City_Coimbatore", "City_Kumbakonam",
            "City_Madurai", "City_Trichy"
        ]

        for col in expected_cols:
            if col not in macro_seq.columns:
                macro_seq[col] = 0

        macro_seq = macro_seq[expected_cols]
        macro_scaled = macro_scaler.transform(macro_seq)
        macro_input = macro_scaled.reshape(1, 12, 15)
        print(f"✅ Macro input ready: {macro_input.shape}")

        # =========================
        # PREDICTION
        # =========================
        print("🚀 Running MiMa LSTM model...")
        predictions = []

        pred = mima_model.predict([micro_input, macro_input], verbose=0)

        for step in range(5):
            pred_step = pred[0, step, :]
            pred_real = scaler_y.inverse_transform([pred_step])[0]
            predictions.append(pred_real)

        pred_real = np.array(predictions)
        print(f"🎯 Model output: {pred_real.round(1)}")

        # =========================
        # FORMAT OUTPUT
        # =========================
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

        print(f"📈 Final forecast: {[(f['temp'], f['hum'], f['wind']) for f in forecast]}")
        print("✅ Prediction complete!\n")

        return render_template("index.html", forecast=forecast)

    except Exception as e:
        print(f"❌ REAL ERROR: {e}")
        return render_template(
            "index.html",
            warning=f"⚠️ Prediction failed: {str(e)}"
        )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
