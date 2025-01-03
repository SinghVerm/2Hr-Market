import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('Market_Direction3_Final_Model.pkl')

# Dropdown options for each feature
previous_day_trend_options = {
    "Select an option": None,
    "Double Top": 1,
    "Double Bottom": 0,
    "Long": 2,
    "No View": 3,
    "Short": 4
}

zone_interaction_options = {
    "Select an option": None,
    "No Interaction": 1,
    "Resistance Above": 2,
    "Resistance Below": 3,
    "Support Above": 4,
    "Support Below": 5,
    "Within Resistance Zone": 6,
    "Within Support Zone": 7
}

prev_hl_options = {
    "Select an option": None,
    "Above": 0,
    "Below": 1,
    "Long": 2,
    "No Touch": 3,
    "Short": 4
}

candle_pattern_options = {
    "Select an option": None,
    "Doji": 0,
    "Green Candle": 1,
    "Green Hammer": 2,
    "Inverted Hammer": 3,
    "Red Candle": 4,
    "Red Hammer": 5
}

# Title
st.title("Market Direction Prediction")

# User Inputs
previous_day_trend = st.selectbox("Previous Day Trend", options=list(previous_day_trend_options.keys()))
zone_interaction = st.selectbox("Zone Interaction", options=list(zone_interaction_options.keys()))
prev_hl = st.selectbox("PrevH/L", options=list(prev_hl_options.keys()))
candle_pattern = st.selectbox("Candle Pattern", options=list(candle_pattern_options.keys()))

# Check for missing values
if None in [
    previous_day_trend_options[previous_day_trend],
    zone_interaction_options[zone_interaction],
    prev_hl_options[prev_hl],
    candle_pattern_options[candle_pattern]
]:
    st.warning("Please select a value for all features before making a prediction.")
else:
    # Encode user inputs into the correct order
    input_features = pd.DataFrame([[
        previous_day_trend_options[previous_day_trend],
        zone_interaction_options[zone_interaction],
        prev_hl_options[prev_hl],
        candle_pattern_options[candle_pattern]
    ]], columns=[
        "Previous_Day_Trend", 
        "Zone Interaction", 
        "PrevH/L", 
        "Candle Pattern"
    ])

    # Predict market direction and probabilities
    prediction = model.predict(input_features)[0]
    probabilities = model.predict_proba(input_features)[0]

    # Calculate confidence score
    score = (probabilities[0] - probabilities[1]) * 100  # Scale to percentage

    # Determine confidence level
    confidence = "Low Confidence" if -60 < score < 60 else "High Confidence"

    # Display results
    prediction_label = "Long" if prediction == 0 else "Short"
    if confidence == "Low Confidence":
        st.subheader(f"{prediction_label} but Low Confidence")
        st.write(f"Score: {score:.2f}")
    else:
        st.subheader(f"Predicted Market Direction: {prediction_label}")
        st.write(f"Score: {score:.2f}")

    # Always display probabilities for Long and Short
    st.write(f"Probability of Long: {probabilities[0] * 100:.2f}%")
    st.write(f"Probability of Short: {probabilities[1] * 100:.2f}%")
