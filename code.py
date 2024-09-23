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
st.title('Heart Disease Prediction')

st.write("Please enter the following patient information:")

# User inputs
cp = st.selectbox('Chest Pain Type (cp)', [0, 1, 2, 3])
thal = st.selectbox('Thalassemia (thal)', [0, 1, 2, 3])
ca = st.number_input('Number of Major Vessels (ca)', 0, 4)
age = st.number_input('Age', 29, 77)
oldpeak = st.number_input('Oldpeak (depression induced by exercise relative to rest)', 0.0, 6.2)
chol = st.number_input('Cholesterol (chol)', 126, 564)

# Collect input data
input_data = {
    'cp': cp,
    'thal': thal,
    'ca': ca,
    'age': age,
    'oldpeak': oldpeak,
    'chol': chol
}

# Predict button
if st.button('Predict'):
    prediction = predict(input_data)
    if prediction == 1:
        st.success("The model predicts: **Heart Disease**")
    else:
        st.success("The model predicts: **No Heart Disease**")
