import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model_filename = 'best_DTC(Heart).joblib'
model = joblib.load(model_filename)

# Define the function to make predictions
def predict(features):
    # Ensure features are in the correct shape for prediction
    input_data = pd.DataFrame([features], columns=['age', 'cp', 'thal', 'ca', 'oldpeak', 'chol'])
    return model.predict(input_data)[0]

# Define the main function for the Streamlit app
def main():
    st.title("Heart Disease Prediction")
    
    # Input fields for the selected features
    age = st.number_input("Age", min_value=0, max_value=120)
    cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", options=[0, 1, 2, 3])
    ca = st.number_input("Number of Major Vessels (0-3)", min_value=0, max_value=3)
    oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=6.0, step=0.1)
    chol = st.number_input("Cholesterol (mg/dl)", min_value=0, max_value=600)

    # Create a button for prediction
    if st.button("Predict"):
        features = [age, cp, thal, ca, oldpeak, chol]
        prediction = predict(features)
        if prediction == 1:
            st.success("The model predicts that you have heart disease.")
        else:
            st.success("The model predicts that you do not have heart disease.")

# Run the app
if __name__ == "__main__":
    main()
