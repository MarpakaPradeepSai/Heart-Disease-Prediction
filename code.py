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
    thal = st.selectbox('Thalassemia (thal)', 
                        options=[(0, 'Normal (0)'), (1, 'Fixed Defect (1)'), (2, 'Reversible Defect (2)')],
                        format_func=lambda x: x[1])
    age = st.number_input('Age', min_value=1, value=None, step=1)  # Only integer input for age

with col3:
    oldpeak = st.number_input('Oldpeak (exercise-induced drop)', min_value=0.0, max_value=6.2, value=0.0)
    chol = st.number_input('Cholesterol (chol)', min_value=126, value=None)  # Allow decimal input for cholesterol

# Collect input data
input_data = {
    'cp': cp,
    'thal': thal[0],  # Get the value from the tuple
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
    # Check if age and chol are filled in
    if age is None or chol is None:
        st.error("⚠️ Please fill in all fields with appropriate values before making a prediction.")
    elif age < 1:
        st.error("⚠️ Age must be at least 1. Please provide a valid age.")
    else:
        # Proceed with prediction if all conditions are met
        prediction = predict(input_data)
        if prediction == 1:
            st.markdown("""
                <div style="background-color:red; padding:20px; text-align:center; border-radius:10px;">
                    <h3 style="color:white;">⚠️ <strong>Alert!</strong></h3>
                    <p style="font-size:18px; color:white;">You have a <strong>high risk</strong> of heart disease.</p>
                    <p style="font-size:18px; color:white;">It's crucial to seek advice from a healthcare professional.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style="background-color:green; padding:20px; text-align:center; border-radius:10px;">
                    <h3 style="color:white;">🎉 <strong>Great News!</strong></h3>
                    <p style="font-size:18px; color:white;">You have a <strong>low risk</strong> of heart disease.</p>
                    <p style="font-size:18px; color:white;">Keep nurturing your heart with healthy choices! 💚</p>
                </div>
            """, unsafe_allow_html=True)
