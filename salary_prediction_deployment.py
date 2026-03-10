import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(page_title="Salary Prediction App", page_icon="💰")

# Load model and encoders
model = joblib.load("salary_prediction_rfr_model.pkl")
encoder = joblib.load("label_encoder_salary.pkl")

# Title
st.title("💰 Salary Prediction Model")

st.write("Enter the details below to predict the salary.")

# User Inputs
age = st.number_input("Enter your Age", min_value=18, max_value=65, value=25)

gender = st.selectbox(
    "Select your Gender",
    encoder["Gender"].classes_
)

education = st.selectbox(
    "Select your Education Level",
    encoder["Education Level"].classes_
)

job_title = st.selectbox(
    "Select your Job Title",
    encoder["Job Title"].classes_
)

experience = st.number_input(
    "Enter Years of Experience",
    min_value=0,
    max_value=50,
    value=1
)

# Create dataframe
df = pd.DataFrame({
    "Age": [age],
    "Gender": [gender],
    "Education Level": [education],
    "Job Title": [job_title],
    "Years of Experience": [experience]
})

# Prediction button
if st.button("Predict Salary"):

    # Encode categorical columns
    for col in encoder:
        if col in df.columns:
            df[col] = encoder[col].transform(df[col])

    # Predict
    prediction = model.predict(df)

    # Output
    st.success(f"💵 Predicted Salary: {prediction[0]:,.2f}")
