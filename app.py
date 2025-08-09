# app.py

import streamlit as st
import pandas as pd
from datetime import datetime
from haversine import haversine, Unit
import joblib

# Use caching to load the model and scaler only once
# This will prevent reloading every time there is user interaction
@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load("model.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except FileNotFoundError:
        st.error("Error: One of the files (model.pkl or scaler.pkl) was not found.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading the model or scaler: {e}")
        st.stop()

# --- Part 1: Load the Model and Scaler ---
model, scaler = load_model_and_scaler()

# --- Part 2: Function for Feature Engineering ---
# This function processes user input into the features required by the model
def create_features(pickup_datetime, pickup_coords, dropoff_coords, passenger_count):
    # Create a dictionary to hold the features
    input_data = {
        "pickup_datetime": pickup_datetime,
        "pickup_latitude": pickup_coords[0],
        "pickup_longitude": pickup_coords[1],
        "dropoff_latitude": dropoff_coords[0],
        "dropoff_longitude": dropoff_coords[1],
        "passenger_count": passenger_count
    }
    
    # Extract time-based features
    input_data["year"] = input_data["pickup_datetime"].year
    input_data["month"] = input_data["pickup_datetime"].month
    input_data["day"] = input_data["pickup_datetime"].day
    input_data["day_of_week"] = input_data["pickup_datetime"].weekday()  # Monday=0, Sunday=6
    input_data["hour"] = input_data["pickup_datetime"].hour

    # Add additional categorical features
    input_data["is_rush_hour"] = 1 if (7 <= input_data["hour"] <= 9 or 16 <= input_data["hour"] <= 19) else 0
    input_data["is_night"] = 1 if (input_data["hour"] < 6 or input_data["hour"] > 22) else 0
    input_data["is_weekend"] = 1 if input_data["day_of_week"] >= 5 else 0

    # Calculate Haversine distance
    input_data["distance_km"] = haversine(pickup_coords, dropoff_coords, unit=Unit.KILOMETERS)

    # Remove the pickup_datetime column and original coordinates as they are not features for the model
    del input_data["pickup_datetime"]
    del input_data["pickup_latitude"]
    del input_data["pickup_longitude"]
    del input_data["dropoff_latitude"]
    del input_data["dropoff_longitude"]

    # Convert the data into a DataFrame format that the model expects
    return pd.DataFrame([input_data])

# --- Part 3: Streamlit UI Components ---
st.title("Uber Fare Prediction")
st.markdown("### Enter your trip details to get a fare estimate.")

with st.form(key="fare_prediction_form"):
    # Date and Time Inputs
    date_input = st.date_input("Select Pickup Date")
    time_input = st.time_input("Select Pickup Time")
    pickup_datetime = datetime.combine(date_input, time_input)

    st.subheader("Pickup Location")
    col1, col2 = st.columns(2)
    with col1:
        pickup_latitude = st.number_input("Pickup Latitude", value=40.7648, format="%f", help="Example: 40.7648")
    with col2:
        pickup_longitude = st.number_input("Pickup Longitude", value=-73.9744, format="%f", help="Example: -73.9744")

    st.subheader("Dropoff Location")
    col3, col4 = st.columns(2)
    with col3:
        dropoff_latitude = st.number_input("Dropoff Latitude", value=40.7573, format="%f", help="Example: 40.7573")
    with col4:
        dropoff_longitude = st.number_input("Dropoff Longitude", value=-74.0044, format="%f", help="Example: -74.0044")

    st.subheader("Passenger Count")
    passenger_count = st.number_input("Number of Passengers", min_value=1, max_value=8, value=1)
    
    submit_button = st.form_submit_button(label="Predict Fare")

# --- Part 4: Prediction Logic ---
if submit_button:
    try:
        # The number_input widget handles the float conversion, so no need for try-except for ValueError here.
        pickup_coords = (pickup_latitude, pickup_longitude)
        dropoff_coords = (dropoff_latitude, dropoff_longitude)
        
        # Create features from user input
        processed_data = create_features(pickup_datetime, pickup_coords, dropoff_coords, passenger_count)
        
        # Define the numeric columns that were scaled during training
        # Note: The original coordinates are no longer here
        numeric_features = [
            "passenger_count", 
            "distance_km", 
            "year", 
            "month", 
            "day", 
            "day_of_week", 
            "hour"
        ]

        # Apply the same scaling transformation
        processed_data[numeric_features] = scaler.transform(processed_data[numeric_features])
        
        # Make the prediction
        prediction = model.predict(processed_data)
        
        # Display the result
        st.success("Prediction Successful!")
        st.metric(label="Estimated Fare", value=f"${prediction[0]:.2f}")
    
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
