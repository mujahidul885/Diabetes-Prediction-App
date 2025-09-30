import os
import joblib
import streamlit as st
import pandas as pd

# Feature names (must match training)
FEATURE_NAMES = [
    "Pregnancies","Glucose","BloodPressure","SkinThickness",
    "Insulin","BMI","DiabetesPedigreeFunction","Age"
]

# Get model path from environment or default
MODEL_PATH = os.getenv("MODEL_PATH", "models/diabeties_model_rf.pkl")

# Try a few candidate paths and raise clear error if not found
candidates = [
    MODEL_PATH,
    os.path.join(os.getcwd(), MODEL_PATH),
    os.path.join(os.getcwd(), "models", "diabeties_model_rf.pkl"),
    "/app/models/diabeties_model_rf.pkl"
]

pipeline = None
for p in candidates:
    try:
        pipeline = joblib.load(p)
        print(f"[INFO] Loaded model from: {p}")
        break
    except FileNotFoundError:
        continue
    except Exception as e:
        st.error(f"Error loading model from {p}: {e}")
        raise

if pipeline is None:
    raise FileNotFoundError(
        "Model file not found. Tried: " + ", ".join(candidates)
    )

st.title("Diabetes Prediction")

inputs = {}
for f in FEATURE_NAMES:
    # you can set defaults smartly; streamlit returns float so cast to float
    inputs[f] = st.number_input(f, value=0.0, format="%.3f")

if st.button("Predict"):
    df = pd.DataFrame([inputs], columns=FEATURE_NAMES)
    pred_class = int(pipeline.predict(df)[0])
    pred_prob = float(pipeline.predict_proba(df)[0,1])
    st.write(f"Predicted class: {pred_class}")
    st.write(f"Probability of diabetes: {pred_prob:.2%}")
