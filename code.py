import streamlit as st
import joblib
import pandas as pd

# Load your best model
model_filename = 'best_DTC(Heart).joblib'
model = joblib.load(model_filename)

# Define the function to make predictions
def predict(features):
    return model.predict([features])[0]

# Streamlit app
st.title("Heart Disease Prediction App 🩺❤️")
st.write("Enter the details below to check if you have heart disease. 📝")

# Input fields for the selected features organized in columns
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=0, max_value=120)

with col2:
    cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3])

with col3:
    thal = st.selectbox("Thalassemia", options=[0, 1, 2, 3])

with col1:
    ca = st.number_input("Number of Major Vessels (0-3)", min_value=0, max_value=3)

with col2:
    oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=6.0, step=0.1)

with col3:
    chol = st.number_input("Cholesterol (mg/dl)", min_value=0, max_value=600)

# Add custom CSS to change button color
st.markdown("""
    <style>
    .stButton > button {
        background-color: #007bff; /* Bootstrap primary blue */
        color: white !important; /* Text color */
        border: none;
        transition: none; /* Remove all transitions */
    }
    .stButton > button:focus,
    .stButton > button:active,
    .stButton > button:hover {
        outline: none; /* Remove focus outline */
        background-color: #007bff !important; /* Keep blue color on focus and active */
        color: white !important; /* Keep text color */
    }
    </style>
    """, unsafe_allow_html=True)

# Prediction button
if st.button("Predict 🔍"):
    features = [age, cp, thal, ca, oldpeak, chol]
    if None in features:
        st.warning("⚠️ Please provide all fields.")
    else:
        prediction = predict(features)
        if prediction == 1:
            st.markdown("""
                <div style="background-color:red; padding:20px; text-align:center; border-radius:10px;">
                    <h3 style="color:white;">⚠️ <strong>Warning!</strong></h3>
                    <p style="font-size:18px; color:white;">You have a <strong>high risk</strong> of heart disease.</p>
                    <p style="color:white;">It's time to take action! Consult a healthcare professional for advice on lifestyle changes.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style="background-color:green; padding:20px; text-align:center; border-radius:10px;">
                    <h3 style="color:white;">✅ <strong>Good News!</strong></h3>
                    <p style="font-size:18px; color:white;">You have a <strong>low risk</strong> of heart disease.</p>
                    <p style="color:white;">Keep up the healthy habits! Stay active and maintain a balanced diet.</p>
                </div>
            """, unsafe_allow_html=True)
