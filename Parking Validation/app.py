import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('parking_xgboost_model.pkl')

# Title of the Streamlit app
st.title("Parking Prediction App")

# User inputs for latitude, longitude, business review count, and is_open status
latitude = st.number_input("Enter Latitude", format="%.6f")
longitude = st.number_input("Enter Longitude", format="%.6f")
business_review_count = st.number_input("Enter Business Review Count", min_value=0, step=1)
is_open = st.selectbox("Is the business open?", [0, 1])

# Predict button
if st.button("Predict Parking Availability"):
    # Prepare the input features for prediction
    input_data = np.array([[latitude, longitude, business_review_count, is_open]])

    # Make a prediction using the model
    prediction = model.predict(input_data)

    # Display the result
    if prediction == 1:
        st.success("Parking is Validated")
    else:
        st.error("Parking is Not Validated")
