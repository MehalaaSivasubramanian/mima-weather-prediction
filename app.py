import streamlit as st
import keras
import numpy as np
import pandas as pd
import joblib
from datetime import timedelta

# ------------------------------------------------
# TITLE
# ------------------------------------------------
st.title("MiMa Weather Prediction System")
st.write("Enter details to predict weather conditions")

# ------------------------------------------------
# USER INPUT
# ------------------------------------------------
city = st.selectbox(
    "Select City",
    ["Trichy","Madurai","Kumbakonam","Chennai","Coimbatore"]
)

date = st.date_input("Select Date")
time = st.time_input("Select Time")

# ------------------------------------------------
# LOAD MODELS
# ------------------------------------------------
@st.cache_resource
def load_models():

    micro_model = keras.models.load_model("micro_model.keras")
    macro_model = keras.models.load_model("macro_model_updated.keras")
    mima_model  = keras.models.load_model("mima_model.keras")

    return micro_model, macro_model, mima_model


micro_model, macro_model, mima_model = load_models()

# ------------------------------------------------
# LOAD DATASETS FROM GOOGLE DRIVE
# ------------------------------------------------
@st.cache_data
def load_data():

    micro_df = pd.read_csv(
        "https://drive.google.com/uc?id=17eeKYcev5Bvw3QooTTLkP-i69hdkqZTo"
    )

    macro_df = pd.read_csv(
        "https://drive.google.com/uc?id=1YyNi7cFLHm2VIei234lpIqC0y64jWdVt"
    )

    micro_df["Datetime"] = pd.to_datetime(
        micro_df[["Year","Month","Day","Hour","Minute"]]
    )

    macro_df["Datetime"] = pd.to_datetime(macro_df["datetime"])

    return micro_df, macro_df


micro_df, macro_df = load_data()

# ------------------------------------------------
# LOAD SCALERS
# ------------------------------------------------
def load_scalers(city):

    city_lower = city.lower()

    micro_X_scaler = joblib.load(f"micro_{city_lower}_scaler_X.pkl")
    micro_y_scaler = joblib.load(f"micro_{city_lower}_scaler_y.pkl")

    macro_X_scaler = joblib.load("macro_X_scaler.pkl")

    return micro_X_scaler, micro_y_scaler, macro_X_scaler


# ------------------------------------------------
# PREDICTION
# ------------------------------------------------
if st.button("Predict Weather"):

    st.write("Preparing data...")

    input_datetime = pd.to_datetime(str(date) + " " + str(time))

    micro_X_scaler, micro_y_scaler, macro_X_scaler = load_scalers(city)

    # ------------------------------------------------
    # MICRO INPUT (24 x 9)
    # ------------------------------------------------
    city_df = micro_df[micro_df["City"] == city]
    city_df = city_df.sort_values("Datetime")

    idx = city_df[city_df["Datetime"] <= input_datetime].index[-1]

    micro_seq = city_df.iloc[idx-24:idx]

    micro_features = micro_seq[
        ["TAIR","RELH","THMP","WSPD","WDIR","WSMX","PRCP","PRES","SRAD"]
    ]

    micro_scaled = micro_X_scaler.transform(micro_features)
    micro_input = micro_scaled.reshape(1,24,9)

    # ------------------------------------------------
    # MACRO INPUT (12 x 15)
    # ------------------------------------------------
    macro_df_sorted = macro_df.sort_values("Datetime")

    idx2 = macro_df_sorted[macro_df_sorted["Datetime"] <= input_datetime].index[-1]

    macro_seq = macro_df_sorted.iloc[idx2-12:idx2].copy()

    macro_seq["City_Chennai"] = (macro_seq["City"]=="Chennai").astype(int)
    macro_seq["City_Coimbatore"] = (macro_seq["City"]=="Coimbatore").astype(int)
    macro_seq["City_Kumbakonam"] = (macro_seq["City"]=="Kumbakonam").astype(int)
    macro_seq["City_Madurai"] = (macro_seq["City"]=="Madurai").astype(int)
    macro_seq["City_Trichy"] = (macro_seq["City"]=="Trichy").astype(int)

    macro_features = macro_seq[
        [
        "ATT1","ATT2","ATT3","ATT4","ATT5",
        "ATT6","ATT7","ATT8","ATT9","ATT10",
        "City_Chennai","City_Coimbatore",
        "City_Kumbakonam","City_Madurai","City_Trichy"
        ]
    ]

    macro_scaled = macro_X_scaler.transform(macro_features)
    macro_input = macro_scaled.reshape(1,12,15)

    # ------------------------------------------------
    # MIMA MODEL PREDICTION
    # ------------------------------------------------
    final_pred_scaled = mima_model.predict([micro_input, macro_input], verbose=0)

    pred_scaled = final_pred_scaled.reshape(-1,3)

    pred_real = micro_y_scaler.inverse_transform(pred_scaled)

    forecast = pred_real.reshape(5,3)

    # ------------------------------------------------
    # PHYSICAL CONSTRAINTS
    # ------------------------------------------------
    forecast[:,2] = np.maximum(forecast[:,2],0)
    forecast[:,1] = np.clip(forecast[:,1],0,100)

    # ------------------------------------------------
    # OUTPUT
    # ------------------------------------------------
    st.subheader("5 Hour Weather Forecast")

    for i in range(5):

        future_time = input_datetime + timedelta(hours=i+1)

        temp = round(forecast[i][0],2)
        hum  = round(forecast[i][1],2)
        wind = round(forecast[i][2],2)

        st.write("Forecast Time:", future_time.strftime("%Y-%m-%d %H:%M"))
        st.write("Temperature:", temp, "°C")
        st.write("Humidity:", hum, "%")
        st.write("Wind Speed:", wind, "km/h")
        st.write("---")
