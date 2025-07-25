import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("model/delay_model.pkl")
scaler = joblib.load("model/scaler.pkl")

st.title("🚌 Sydney Bus Delay Predictor")

st.markdown("Predict whether a Sydney bus will be delayed at a given stop.")

# --- User Inputs ---
stop_sequence = st.number_input("Stop Sequence (e.g., 1, 2, ...)", min_value=1, max_value=100, value=5)

stop_lat = st.number_input("Stop Latitude (e.g., -33.8700)", format="%.6f")
stop_lon = st.number_input("Stop Longitude (e.g., 151.2100)", format="%.6f")

hour_of_day = st.slider("Hour of Day (0-23)", min_value=0, max_value=23, value=8)

day_of_week = st.selectbox("Day of Week", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2,
           'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
dow = day_map[day_of_week]

# --- Build Input ---
input_data = pd.DataFrame([{
    'stop_sequence': stop_sequence,
    'stop_lat': stop_lat,
    'stop_lon': stop_lon,
    'hour_of_day': hour_of_day,
    'day_of_week': dow
}])

# --- Scale and Predict ---
input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)[0]
proba = model.predict_proba(input_scaled)[0][1]

# --- Output ---
if prediction == 1:
    st.error(f"🚨 Likely to be Delayed ({proba:.2%} probability)")
else:
    st.success(f"✅ Likely to be On Time ({1 - proba:.2%} probability)")

st.markdown("Model trained on static GTFS & real-time transit feed data.")
