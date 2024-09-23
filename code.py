# Create a Streamlit app (save this as app.py)
import streamlit as st
import pandas as pd
import joblib

# Load the saved model and features
model = joblib.load('heart_disease_model.joblib')
best_features = joblib.load('best_features.joblib')

def predict_heart_disease(features):
    prediction = model.predict(pd.DataFrame([features], columns=best_features))
    return prediction[0]

st.title('Heart Disease Prediction App')

# Create input fields for each feature
feature_inputs = {}
for feature in best_features:
    feature_inputs[feature] = st.number_input(f'Enter {feature}:', value=0.0)

if st.button('Predict'):
    features = [feature_inputs[feature] for feature in best_features]
    prediction = predict_heart_disease(features)
    st.write('Prediction: ', 'Heart Disease' if prediction == 1 else 'No Heart Disease')
