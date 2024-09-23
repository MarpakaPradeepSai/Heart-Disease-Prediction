import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the saved model and features
try:
    model = joblib.load('heart_disease_model.joblib')
    best_features = joblib.load('best_features.joblib')
except FileNotFoundError:
    st.error("Model files not found. Please ensure the model is trained and saved correctly.")
    st.stop()

def predict_heart_disease(features):
    try:
        # Convert input to float and reshape
        features_array = np.array(features).astype(float).reshape(1, -1)
        # Create DataFrame with correct column names
        df = pd.DataFrame(features_array, columns=best_features)
        prediction = model.predict(df)
        return prediction[0]
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
        return None

st.title('Heart Disease Prediction App')

# Create input fields for each feature
feature_inputs = {}
for feature in best_features:
    feature_inputs[feature] = st.number_input(f'Enter {feature}:', value=0.0, step=0.1)

if st.button('Predict'):
    features = [feature_inputs[feature] for feature in best_features]
    prediction = predict_heart_disease(features)
    if prediction is not None:
        st.write('Prediction: ', 'Heart Disease' if prediction == 1 else 'No Heart Disease')
