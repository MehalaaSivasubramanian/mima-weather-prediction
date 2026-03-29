from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
import keras
import requests
import os
from datetime import timedelta

app = Flask(__name__)

# DOWNLOAD micro.csv FROM YOUR GOOGLE DRIVE LINK
def download_micro_csv():
    """Download YOUR 38MB micro.csv from Google Drive"""
    if os.path.exists('dataset/micro.csv'):
        print("✅ micro.csv already cached!")
        return
    
    print("📥 Downloading micro.csv from Google Drive...")
    FILE_ID = "17eeKYcev5Bvw3QooTTLkP-i69hdkqZTo"  # YOUR LINK
    URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
    
    response = requests.get(URL)
    os.makedirs('dataset', exist_ok=True)
    with open('dataset/micro.csv', 'wb') as f:
        f.write(response.content)
    print("✅ micro.csv downloaded & cached!")

# Download ONCE on startup
download_micro_csv()

# LOAD MODEL & SCALERS (upload these small files to GitHub)
try:
    mima_model = keras.models.load_model("models/mima_model.keras")
    macro_scaler = joblib.load("scalers/macro_X_scaler.pkl")
    print("✅ LSTM model & scalers loaded!")
except:
    print("⚠️ Model missing - using fallback")
    mima_model = macro_scaler = None

# GLOBAL DATA LOAD (performance fix)
MICRO_DF, MACRO_DF = None, None

def load_data():
    global MICRO_DF, MACRO_DF
    if MICRO_DF is None or MACRO_DF is None:
        print("Loading datasets...")
        
        # micro.csv from Google Drive (auto-downloaded)
        MICRO_DF = pd.read_csv("dataset/micro.csv")
        MICRO_DF["Datetime"] = pd.to_datetime(MICRO_DF[["Year","Month","Day","Hour","Minute"]])
        
        # macro.csv fallback (small synthetic data)
        if os.path.exists("dataset/macro.csv"):
            MACRO_DF = pd.read_csv("dataset/macro.csv")
        else:
            print("⚠️ macro.csv missing - creating synthetic...")
            dates = pd.date_range('2020-01-01', periods=10000, freq='H')
            MACRO_DF = pd.DataFrame({
                'datetime': dates,
                'ATT1': np.random.uniform(20, 35, 10000),
                'ATT2': np.random.uniform(60, 90, 10000),
                'ATT3': np.random.uniform(5, 15, 10000),
                'ATT4': np.random.uniform(0, 10, 10000),
                'ATT5': np.random.uniform(990, 1020, 10000),
                'ATT6': np.random.uniform(0, 1000, 10000),
                'ATT7': np.random.uniform(0, 50, 10000),
                'ATT8': np.random.uniform(0, 360, 10000),
                'ATT9': np.random.uniform(0, 20, 10000),
                'ATT10': np.random.uniform(0, 5, 10000),
                'City': np.random.choice(['Chennai', 'Coimbatore', 'Madurai', 'Trichy', 'Kumbakonam'], 10000)
            })
            MACRO_DF["Datetime"] = pd.to_datetime(MACRO_DF["datetime"])
        
        print("✅ Datasets loaded globally")
    return MICRO_DF, MACRO_DF

def fix_values(temp, hum, wind, city):
    if city == "Chennai": temp = np.clip(temp, 26, 40)
    elif city == "Coimbatore": temp = np.clip(temp, 20, 32)
    elif city == "Madurai": temp = np.clip(temp, 25, 39)
    elif city == "Trichy": temp = np.clip(temp, 26, 40)
    else: temp = np.clip(temp, 24, 38)
    hum = np.clip(hum, 50, 90)
    wind = np.clip(hum, 2, 20)
    return round(temp, 1), round(hum, 1), round(wind, 1)

def fallback_weather(city):
    base_temp = {"Chennai":32, "Madurai":31, "Trichy":31, "Kumbakonam":30, "Coimbatore":27}.get(city, 30)
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
        input_datetime = pd.to_datetime(date + " " + time)
        
        # 🔥 **STRICT 2021-2024 VALIDATION**
        min_date = pd.Timestamp("2021-01-01 00:00")
        max_date = pd.Timestamp("2024-12-31 23:59")
        
        if not (min_date <= input_datetime <= max_date):
            return render_template("index.html", 
                                 warning=f"⚠️ Date must be between 2021-01-01 and 2024-12-31. Selected: {input_datetime.strftime('%Y-%m-%d %H:%M')}")
        
        print(f"\n🔍 PREDICTING {city} at {input_datetime}")
        
        micro_df, macro_df = load_data()
        
        # === MICRO DATA (48 timesteps REQUIRED) ===
        city_micro = micro_df[micro_df["City"] == city].sort_values("Datetime")
        valid_micro = city_micro[city_micro["Datetime"] <= input_datetime]
        
        print(f"📊 Micro: {len(city_micro)} total, {len(valid_micro)} <= input")
        print(f"⏰ Micro range: {valid_micro['Datetime'].min()} to {valid_micro['Datetime'].max()}")
        
        use_model = len(valid_micro) >= 48
        
        if use_model and mima_model is not None:
            micro_seq = valid_micro.tail(48)
            features = ["TAIR","RELH","THMP","WSPD","WDIR","WSMX","PRCP","PRES","SRAD"]
            micro_features = micro_seq[features]
            
            scaler_X = joblib.load(f"scalers/micro_{city.lower()}_scaler_X.pkl")
            micro_scaled = scaler_X.transform(micro_features)
            micro_input = micro_scaled.reshape(1, 48, 9)
            print(f"✅ Micro input ready: {micro_input.shape}")
        else:
            print("❌ Insufficient micro data or model - using fallback")
            micro_input = None

        # === MACRO DATA ===
        valid_macro = macro_df[macro_df["Datetime"] <= input_datetime].sort_values("Datetime")
        print(f"📊 Macro: {len(valid_macro)} rows <= input")
        
        if len(valid_macro) >= 12 and macro_scaler is not None:
            macro_seq = valid_macro.tail(12).copy()
            macro_seq = macro_seq[["ATT1","ATT2","ATT3","ATT4","ATT5","ATT6","ATT7","ATT8","ATT9","ATT10","City"]]
            macro_seq = pd.get_dummies(macro_seq, columns=["City"])
            
            expected_cols = ["ATT1","ATT2","ATT3","ATT4","ATT5","ATT6","ATT7","ATT8","ATT9","ATT10",
                           "City_Chennai","City_Coimbatore","City_Kumbakonam","City_Madurai","City_Trichy"]
            for col in expected_cols:
                if col not in macro_seq.columns:
                    macro_seq[col] = 0
            macro_seq = macro_seq[expected_cols]
            
            macro_scaled = macro_scaler.transform(macro_seq)
            macro_input = macro_scaled.reshape(1, 12, 15)
            print(f"✅ Macro input ready: {macro_input.shape}")
        else:
            print("❌ Insufficient macro data - using fallback")
            macro_input = None

        # === PREDICTION ===
        if use_model and micro_input is not None and macro_input is not None:
            print("🚀 Running MiMa LSTM model...")
            scaler_y = joblib.load(f"scalers/micro_{city.lower()}_scaler_y.pkl")
            predictions = []
            current_micro = micro_input.copy()
            
            for step in range(5):
                pred = mima_model.predict([current_micro, macro_input], verbose=0)
                pred_step = pred[0, step, :]  # Shape: (3,)
                
                pred_real = scaler_y.inverse_transform([pred_step])[0]
                predictions.append(pred_real)
                
                # Rolling window update
                next_row = current_micro[0, -1, :].copy()
                next_row[0] = pred_step[0]  # TAIR
                next_row[1] = pred_step[1]  # RELH
                next_row[3] = pred_step[2]  # WSPD
                current_micro = np.roll(current_micro, -1, axis=1)
                current_micro[0, -1, :] = next_row
            
            pred_real = np.array(predictions)
            print(f"🎯 Model output: {pred_real.round(1)}")
        else:
            pred_real = fallback_weather(city)
            print("🔄 Using fallback predictions")

        # === FORMAT OUTPUT ===
        forecast = []
        for i in range(5):
            future_time = input_datetime + timedelta(hours=i+1)
            temp, hum, wind = fix_values(pred_real[i][0], pred_real[i][1], pred_real[i][2], city)
            forecast.append({
                "time": future_time.strftime("%Y-%m-%d %H:%M"),
                "temp": temp, "hum": hum, "wind": wind
            })
        
        print(f"📈 Final forecast: {[(f['temp'],f['hum'],f['wind']) for f in forecast]}")
        print("✅ Prediction complete!\n")
        
        return render_template("index.html", forecast=forecast)
    
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return render_template("index.html", warning=f"⚠️ Prediction failed: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
