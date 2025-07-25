import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

# Load model and scaler
model = joblib.load("model/delay_model.pkl")
scaler = joblib.load("model/scaler.pkl")

st.set_page_config(page_title="Sydney Bus Delay Predictor", layout="centered")

st.title("🚌 Sydney Bus Delay Prediction")
st.markdown("Predict whether a bus is delayed at a particular stop based on schedule and simulated conditions.")

# User Inputs
trip_id = st.text_input("Trip ID", "1.AA51.1-SC0-1-sj2-3.1.R")
stop_sequence = st.number_input("Stop Sequence", min_value=1, value=3)
scheduled_arrival = st.time_input("Scheduled Arrival Time", value=datetime.now().time())

# Simulated delay input (in minutes)
simulated_delay = st.slider("Simulated Delay (min)", 0, 30, 5)

if st.button("Predict Delay Status"):
    try:
        # Convert time to seconds from midnight
        scheduled_time = datetime.combine(datetime.today(), scheduled_arrival)
        actual_time = scheduled_time + timedelta(minutes=simulated_delay)

        sched_seconds = scheduled_time.hour * 3600 + scheduled_time.minute * 60 + scheduled_time.second
        actual_seconds = actual_time.hour * 3600 + actual_time.minute * 60 + actual_time.second

        delay_minutes = (actual_seconds - sched_seconds) / 60

        # Prepare model input
        input_df = pd.DataFrame([{
            'stop_sequence': stop_sequence,
            'scheduled_time': sched_seconds,
            'actual_time': actual_seconds,
            'delay_minutes': delay_minutes
        }])

        # Scale
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0][1]

        status = "🟢 On Time" if prediction == 0 else "🔴 Delayed"
        st.markdown(f"### Prediction: {status}")
        st.metric(label="Delay Probability", value=f"{proba:.2%}")

    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")
