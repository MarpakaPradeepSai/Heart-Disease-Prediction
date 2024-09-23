import streamlit as st
import pandas as pd
import joblib

# Load the model
model = joblib.load('best_random_forest_model.pkl')

# Function to make predictions
def predict(input_data):
    prediction = model.predict(input_data)
    return prediction[0]

# Streamlit UI
st.title('Diabetes Prediction App')

# Input fields for each feature
age = st.number_input('Age', min_value=0)
cp = st.number_input('Chest Pain Type (0-3)', min_value=0, max_value=3)
ca = st.number_input('Number of Major Vessels (0-3)', min_value=0, max_value=3)
thalach = st.number_input('Maximum Heart Rate Achieved', min_value=0)
oldpeak = st.number_input('Depression Induced by Exercise (in units)', min_value=0.0)
thal = st.number_input('Thalassemia (1-3)', min_value=1, max_value=3)

# Create a DataFrame for the input data
input_data = pd.DataFrame({
    'age': [age],
    'cp': [cp],
    'ca': [ca],
    'thalach': [thalach],
    'oldpeak': [oldpeak],
    'thal': [thal]
})

# Make prediction when button is pressed
if st.button('Predict'):
    result = predict(input_data)
    st.write(f'Prediction: {"Diabetes" if result == 1 else "No Diabetes"}')

