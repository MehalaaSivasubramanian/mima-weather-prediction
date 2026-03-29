@app.route("/predict", methods=["POST"])
def predict():
    try:
        city = request.form["city"]
        date = request.form["date"]
        time = request.form["time"]
        input_datetime = pd.to_datetime(date + " " + time)

        print(f"\n🔍 PREDICTING {city} at {input_datetime}")

        # =========================
        # STRICT YEAR VALIDATION - ONLY 2021, 2022, 2023 ALLOWED
        # =========================
        input_year = input_datetime.year
        
        if input_year not in [2021, 2022, 2023]:
            warning = f"❌ ERROR: Dataset only has data for **2021-2023** only! Year **{input_year}** is NOT supported."
            print(f"🚫 BLOCKED YEAR {input_year}")
            return render_template("index.html", warning=warning)
        
        print(f"✅ VALID YEAR {input_year} - Processing...")
        
        # [ALL EXISTING PREDICTION LOGIC - SAME AS BEFORE]
        micro_df, macro_df = load_data()
        
        city_micro = micro_df[micro_df["City"] == city].sort_values("Datetime")
        valid_micro = city_micro[city_micro["Datetime"] <= input_datetime]
        print(f"📊 Micro: {len(city_micro)} total, {len(valid_micro)} <= input")

        use_model = len(valid_micro) >= 48

        if use_model and mima_model is not None:
            micro_seq = valid_micro.tail(48)
            features = ["TAIR", "RELH", "THMP", "WSPD", "WDIR", "WSMX", "PRCP", "PRES", "SRAD"]
            micro_features = micro_seq[features]

            scaler_X = joblib.load(f"micro_{city.lower()}_scaler_X.pkl")
            micro_scaled = scaler_X.transform(micro_features)
            micro_input = micro_scaled.reshape(1, 48, 9)
            print(f"✅ Micro input ready: {micro_input.shape}")
        else:
            print("❌ Insufficient micro data or model - using fallback")
            micro_input = None

        valid_macro = macro_df[macro_df["Datetime"] <= input_datetime].sort_values("Datetime")
        print(f"📊 Macro: {len(valid_macro)} rows <= input")

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
            print(f"✅ Macro input ready: {macro_input.shape}")
        else:
            print("❌ Insufficient macro data - using fallback")
            macro_input = None

        if use_model and micro_input is not None and macro_input is not None:
            print("🚀 Running MiMa LSTM model...")
            scaler_y = joblib.load(f"micro_{city.lower()}_scaler_y.pkl")
            predictions = []

            pred = mima_model.predict([micro_input, macro_input], verbose=0)

            for step in range(5):
                pred_step = pred[0, step, :]
                pred_real = scaler_y.inverse_transform([pred_step])[0]
                predictions.append(pred_real)

            pred_real = np.array(predictions)
            print(f"🎯 Model output: {pred_real.round(1)}")
        else:
            pred_real = fallback_weather(city)
            print("🔄 Using fallback predictions")

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
        print(f"❌ ERROR: {e}")
        warning = "⚠️ Prediction failed. Please check your input and try again."
        return render_template("index.html", warning=warning)
