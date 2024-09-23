import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('best_decision_tree_model.pkl')

# Top features identified
best_features = ['cp', 'thal', 'ca', 'age', 'oldpeak', 'chol']

# Function to make predictions
def predict(features):
    return model.predict([features])

# Streamlit app layout
st.title("Heart Disease Prediction")
st.write("Enter the details below:")

# Input fields for the best features
cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3])
thal = st.selectbox("Thalassemia", options=[0, 1, 2, 3])
ca = st.selectbox("Number of Major Vessels (0-3) Colored by Fluoroscopy", options=[0, 1, 2, 3])
age = st.number_input("Age", min_value=1, max_value=120)
oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=6.0)
chol = st.number_input("Cholesterol", min_value=100, max_value=600)

# Button to predict
if st.button("Predict"):
    features = [cp, thal, ca, age, oldpeak, chol]
    prediction = predict(features)
    
    # Show the prediction result
    if prediction[0] == 1:
        st.success("The model predicts: Heart Disease Present")
    else:
        st.success("The model predicts: No Heart Disease")
