import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('best_DTC(Heart).pkl')

# Define the feature names
feature_names = ['cp', 'thal', 'ca', 'age', 'oldpeak', 'chol']

# Function to make predictions
def predict(input_data):
    data = pd.DataFrame([input_data], columns=feature_names)
    prediction = model.predict(data)
    return prediction[0]

# Streamlit app
st.title('Heart Disease Prediction App 🫀')
st.write("Enter the details below to check your heart disease risk. 📝")

# Input fields for the features organized in columns
col1, col2, col3 = st.columns(3)

with col1:
    cp = st.selectbox('Chest Pain Type (cp)', [0, 1, 2, 3])
    ca = st.selectbox('Number of Major Vessels (ca)', [0, 1, 2, 3, 4])

with col2:
    thal = st.selectbox('Thalassemia (thal)', [0, 1, 2, 3])
    age = st.number_input('Age', min_value=1, value=1)  # Minimum age is set to 1

with col3:
    oldpeak = st.number_input('Oldpeak (exercise-induced drop)', min_value=0.1, max_value=6.2, value=0.1)  # Minimum set to 0.1
    chol = st.number_input('Cholesterol (chol)', min_value=127, max_value=564, value=127)  # Minimum set to 127

# Collect input data
input_data = {
    'cp': cp,
    'thal': thal,
    'ca': ca,
    'age': age,
    'oldpeak': oldpeak,
    'chol': chol
}

# Add custom CSS to change button color without hover or active effect
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

# Predict button
if st.button('Predict 🔍'):
    # Check if all required fields are filled
    if age < 1 or oldpeak < 0.1 or chol < 127:
        st.error("⚠️ Please fill in all fields with appropriate values before making a prediction.")
    else:
        prediction = predict(input_data)
        if prediction == 1:
            st.markdown("""
                <div style="background-color:red; padding:20px; text-align:center; border-radius:10px;">
                    <h3 style="color:white;">⚠️ <strong>Alert!</strong></h3>
                    <p style="font-size:20px; color:white;">The model indicates a potential <strong>Heart Disease</strong> risk.</p>
                    <p style="color:white;">It's crucial to seek advice from a healthcare professional.</p>
                    <img src="https://example.com/heart_warning.png" style="width:50%; max-width:300px;" alt="Heart Warning">
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style="background-color:green; padding:20px; text-align:center; border-radius:10px;">
                    <h3 style="color:white;">🎉 <strong>Great News!</strong></h3>
                    <p style="font-size:20px; color:white;">The model predicts: <strong>No Heart Disease</strong></p>
                    <p style="color:white;">Keep nurturing your heart with healthy choices! 💚</p>
                    <img src="https://example.com/heart_healthy.png" style="width:50%; max-width:300px;" alt="Heart Healthy">
                </div>
            """, unsafe_allow_html=True)
