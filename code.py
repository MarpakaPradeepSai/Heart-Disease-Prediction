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
    ca = st.number_input('Number of Major Vessels (ca)', 0, 4)

with col2:
    thal = st.selectbox('Thalassemia (thal)', [0, 1, 2, 3])
    age = st.number_input('Age')

with col3:
    oldpeak = st.number_input('Oldpeak (exercise-induced drop)'), min_value=0.0, max_value=6.2, value=None)
    chol = st.number_input('Cholesterol (chol)', min_value=126, max_value=564, value=None)

# Collect input data
input_data = {
    'cp': cp,
    'thal': thal,
    'ca': ca,
    'age': age,
    'oldpeak': oldpeak if oldpeak is not None else 0.0,  # Handle None
    'chol': chol if chol is not None else 126  # Handle None
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
    prediction = predict(input_data)
    if prediction == 1:
        st.markdown("""
            <div style="background-color:red; padding:20px; text-align:center; border-radius:10px;">
                <h3 style="color:white;">⚠️ <strong>Warning!</strong></h3>
                <p style="font-size:18px; color:white;">The model predicts: <strong>Heart Disease</strong></p>
                <p style="color:white;">Please consult a healthcare professional for advice.</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style="background-color:green; padding:20px; text-align:center; border-radius:10px;">
                <h3 style="color:white;">✅ <strong>Good News!</strong></h3>
                <p style="font-size:18px; color:white;">The model predicts: <strong>No Heart Disease</strong></p>
                <p style="color:white;">Keep up the healthy habits!</p>
            </div>
        """, unsafe_allow_html=True)
