from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import os
import requests
from datetime import timedelta
from tensorflow.keras.models import load_model

app = Flask(__name__)

# -------------------------
# Globals
# -------------------------
mima_model = None
macro_scaler = None
micro_df = None
macro_df = None

# -------------------------
# Download micro.csv from Google Drive
# -------------------------
def download_file(file_id, destination):
    if os.path.exists(destination):
        print(f"✅ {destination} already exists")
        return

    print(f"⬇️ Downloading {destination}...")

    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    session = requests.Session()
    response = session.get(url, stream=True)

    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            url = f"https://drive.google.com/uc?export=download&confirm={value}&id={file_id}"
            response = session.get(url, stream=True)
            break

    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

    print("✅ Download complete")

# -------------------------
# Load Data
# -------------------------
def load_data():
    global micro_df, macro_df

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    micro_path = os.path.join(BASE_DIR, "micro.csv")
    macro_path = os.path.join(BASE_DIR, "macro.csv")

    MICRO_ID = "1YyNi7cFLHm2VIei234lpIqC0y64jWdVt"

    # Download micro.csv from Drive
    download_file(MICRO_ID, micro_path)

    # Read files
    micro_df = pd.read_csv(micro_path)
    macro_df = pd.read_csv(macro_path)

    # Clean column names
    micro_df.columns = micro_df.columns.str.strip()
    macro_df.columns = macro_df.columns.str.strip()

    print("📌 MICRO COLUMNS:", list(micro_df.columns))
    print("📌 MACRO COLUMNS:", list(macro_df.columns))

    # -------------------------
    # MICRO: Create Datetime
    # -------------------------
    micro_df["Datetime"] = pd.to_datetime(
        micro_df[["Year", "Month", "Day", "Hour", "Minute"]]
    )

    # -------------------------
    # MACRO: Rename datetime -> Datetime
    # -------------------------
    if "datetime" in macro_df.columns:
        macro_df.rename(columns={"datetime": "Datetime"}, inplace=True)

    macro_df["Datetime"] = pd.to_datetime(macro_df["Datetime"], dayfirst=True)

    print("✅ Data loaded successfully")

# -------------------------
# Load Models
# -------------------------
def load_models():
    global mima_model, macro_scaler

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    try:
        mima_model = load_model(os.path.join(BASE_DIR, "mima_model.keras"))
        macro_scaler = joblib.load(os.path.join(BASE_DIR, "macro_X_scaler.pkl"))

        print("✅ Models loaded successfully")

    except Exception as e:
        print(f"❌ Model load error: {e}")
        mima_model = None
        macro_scaler = None

# -------------------------
# Fallback
# -------------------------
def fallback_weather():
    return np.array([
        [30, 70, 10],
        [31, 68, 11],
        [29, 72, 9],
        [28, 75, 8],
        [27, 78, 7]
    ])

# -------------------------
# Fix output values
# -------------------------
def fix_values(temp, hum, wind):
    temp = round(float(temp), 1)
    hum = round(float(hum), 1)
    wind = round(float(wind), 1)

    temp = max(min(temp, 50), -10)
    hum = max(min(hum, 100), 0)
    wind = max(min(wind, 150), 0)

    return temp, hum, wind

# -------------------------
# Initialize once
# -------------------------
load_data()
load_models()

# -------------------------
# Routes
# -------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        city = request.form["city"]
        date = request.form["date"]
        time = request.form["time"]

        input_datetime = pd.to_datetime(date + " " + time)

        print(f"\n🔍 PREDICTING {city} at {input_datetime}")

        # -------------------------
        # YEAR VALIDATION
        # -------------------------
        input_year = input_datetime.year

        if input_year > 2023 or input_year < 2021:
            warning = f"❌ ERROR: Dataset has data for 2021-2024, but predictions are supported only for 2021-2023. Year {input_year} is NOT supported."
            print(f"🚫 BLOCKED YEAR {input_year}")
            return render_template("index.html", warning=warning)

        print(f"✅ VALID YEAR {input_year} - Processing...")

        # -------------------------
        # City mapping for scaler files
        # -------------------------
        city_map = {
            "Chennai": "chennai",
            "Coimbatore": "cbe",
            "Kumbakonam": "kumbakonam",
            "Madurai": "madurai",
            "Trichy": "trichy"
        }

        city_key = city_map[city]
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        # -------------------------
        # MICRO INPUT
        # -------------------------
        city_micro = micro_df[micro_df["City"] == city].sort_values("Datetime")
        valid_micro = city_micro[city_micro["Datetime"] <= input_datetime]

        print(f"📊 Micro rows available: {len(valid_micro)}")

        if len(valid_micro) >= 48 and mima_model is not None:
            micro_seq = valid_micro.tail(48)

            features = ["TAIR", "RELH", "THMP", "WSPD", "WDIR", "WSMX", "PRCP", "PRES", "SRAD"]
            micro_features = micro_seq[features]

            scaler_X = joblib.load(os.path.join(BASE_DIR, f"micro_{city_key}_scaler_X.pkl"))
            scaler_y = joblib.load(os.path.join(BASE_DIR, f"micro_{city_key}_scaler_y.pkl"))

            micro_scaled = scaler_X.transform(micro_features)
            micro_input = micro_scaled.reshape(1, 48, 9)

            print(f"✅ Micro input shape: {micro_input.shape}")
        else:
            micro_input = None
            scaler_y = None
            print("⚠️ Not enough micro data or model missing")

        # -------------------------
        # MACRO INPUT
        # -------------------------
        valid_macro = macro_df[macro_df["Datetime"] <= input_datetime].sort_values("Datetime")

        print(f"📊 Macro rows available: {len(valid_macro)}")

        if len(valid_macro) >= 12 and macro_scaler is not None:
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

            print(f"✅ Macro input shape: {macro_input.shape}")
        else:
            macro_input = None
            print("⚠️ Not enough macro data or scaler missing")

        # -------------------------
        # Prediction
        # -------------------------
        if micro_input is not None and macro_input is not None and scaler_y is not None:
            print("🚀 Running MiMa model...")

            pred = mima_model.predict([micro_input, macro_input], verbose=0)

            predictions = []
            for i in range(5):
                step_pred = pred[0, i, :]
                real_pred = scaler_y.inverse_transform([step_pred])[0]
                predictions.append(real_pred)

            pred_real = np.array(predictions)
            print(f"🎯 Prediction output: {pred_real.round(2)}")
        else:
            pred_real = fallback_weather()
            print("🔄 Using fallback predictions")

        # -------------------------
        # Format forecast
        # -------------------------
        forecast = []
        for i in range(5):
            future_time = input_datetime + timedelta(hours=i + 1)
            temp, hum, wind = fix_values(pred_real[i][0], pred_real[i][1], pred_real[i][2])

            forecast.append({
                "time": future_time.strftime("%Y-%m-%d %H:%M"),
                "temp": temp,
                "hum": hum,
                "wind": wind
            })

        print("✅ Prediction complete")
        return render_template("index.html", forecast=forecast)

    except Exception as e:
        print(f"❌ ERROR: {e}")
        return render_template("index.html", warning=f"⚠️ Prediction failed: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
