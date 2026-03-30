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

    download_file(MICRO_ID, micro_path)

    micro_df = pd.read_csv(micro_path)
    macro_df = pd.read_csv(macro_path)

    micro_df["Datetime"] = pd.to_datetime(micro_df["Datetime"])
    macro_df["Datetime"] = pd.to_datetime(macro_df["Datetime"])

    print("✅ Data loaded")

# -------------------------
# Load Models
# -------------------------
def load_models():
    global mima_model, macro_scaler

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    try:
        mima_model = load_model(os.path.join(BASE_DIR, "mima_model.keras"))
        macro_scaler = joblib.load(os.path.join(BASE_DIR, "macro_X_scaler.pkl"))

        print("✅ Models loaded")

    except Exception as e:
        print("❌ Model load error:", e)
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

def fix_values(t, h, w):
    return round(t,1), round(h,1), round(w,1)

# -------------------------
# Init
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

        # -------------------------
        # YEAR VALIDATION
        # -------------------------
        year = input_datetime.year

        if year > 2023 or year < 2021:
            warning = f"❌ ERROR: Dataset has data for 2021-2024, but predictions are supported only for 2021-2023. Year {year} is NOT supported."
            return render_template("index.html", warning=warning)

        # -------------------------
        # City mapping
        # -------------------------
        city_map = {
            "Chennai": "chennai",
            "Coimbatore": "cbe",
            "Kumbakonam": "kumbakonam",
            "Madurai": "madurai",
            "Trichy": "trichy"
        }

        key = city_map[city]

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        # -------------------------
        # MICRO
        # -------------------------
        df_micro = micro_df[micro_df["City"] == city].sort_values("Datetime")
        valid_micro = df_micro[df_micro["Datetime"] <= input_datetime]

        if len(valid_micro) >= 48 and mima_model:
            seq = valid_micro.tail(48)

            features = ["TAIR","RELH","THMP","WSPD","WDIR","WSMX","PRCP","PRES","SRAD"]
            X = seq[features]

            scaler_X = joblib.load(os.path.join(BASE_DIR, f"micro_{key}_scaler_X.pkl"))
            scaler_y = joblib.load(os.path.join(BASE_DIR, f"micro_{key}_scaler_y.pkl"))

            X_scaled = scaler_X.transform(X)
            micro_input = X_scaled.reshape(1,48,9)
        else:
            micro_input = None

        # -------------------------
        # MACRO
        # -------------------------
        valid_macro = macro_df[macro_df["Datetime"] <= input_datetime]

        if len(valid_macro) >= 12 and macro_scaler:
            seq = valid_macro.tail(12)

            seq = seq[[
                "ATT1","ATT2","ATT3","ATT4","ATT5",
                "ATT6","ATT7","ATT8","ATT9","ATT10","City"
            ]]

            seq = pd.get_dummies(seq, columns=["City"])

            expected = [
                "ATT1","ATT2","ATT3","ATT4","ATT5",
                "ATT6","ATT7","ATT8","ATT9","ATT10",
                "City_Chennai","City_Coimbatore",
                "City_Kumbakonam","City_Madurai","City_Trichy"
            ]

            for col in expected:
                if col not in seq.columns:
                    seq[col] = 0

            seq = seq[expected]

            macro_input = macro_scaler.transform(seq).reshape(1,12,15)
        else:
            macro_input = None

        # -------------------------
        # PREDICTION
        # -------------------------
        if micro_input is not None and macro_input is not None:
            pred = mima_model.predict([micro_input, macro_input], verbose=0)

            preds = []
            for i in range(5):
                p = scaler_y.inverse_transform([pred[0,i]])[0]
                preds.append(p)

            pred_real = np.array(preds)
        else:
            pred_real = fallback_weather()

        # -------------------------
        # FORMAT OUTPUT
        # -------------------------
        forecast = []
        for i in range(5):
            future = input_datetime + timedelta(hours=i+1)
            t,h,w = fix_values(pred_real[i][0], pred_real[i][1], pred_real[i][2])

            forecast.append({
                "time": future.strftime("%Y-%m-%d %H:%M"),
                "temp": t,
                "hum": h,
                "wind": w
            })

        return render_template("index.html", forecast=forecast)

    except Exception as e:
        return render_template("index.html", warning=str(e))


if __name__ == "__main__":
    app.run(debug=True)
