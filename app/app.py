import streamlit as st
import pandas as pd
import joblib

# Load pipeline
pipeline = joblib.load("./models/diabeties_model_rf.pkl")

FEATURE_NAMES = ["Pregnancies","Glucose","BloodPressure","SkinThickness",
                 "Insulin","BMI","DiabetesPedigreeFunction","Age"]

st.title("Diabetes Prediction App")

# Input form
inputs = {}
for f in FEATURE_NAMES:
    inputs[f] = st.number_input(f, value=0.0)

if st.button("Predict"):
    df = pd.DataFrame([inputs], columns=FEATURE_NAMES)
    pred_class = int(pipeline.predict(df)[0])
    pred_prob = float(pipeline.predict_proba(df)[0,1])
    st.write(f"Predicted class: {pred_class}")
    st.write(f"Probability of diabetes: {pred_prob:.2%}")
