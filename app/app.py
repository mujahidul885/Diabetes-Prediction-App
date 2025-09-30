# import streamlit as st
# import pandas as pd
# import joblib

# # Load pipeline
# pipeline = joblib.load("./models/diabeties_model_rf.pkl")

# FEATURE_NAMES = ["Pregnancies","Glucose","BloodPressure","SkinThickness",
#                  "Insulin","BMI","DiabetesPedigreeFunction","Age"]

# st.title("Diabetes Prediction App")

# # Input form
# inputs = {}
# for f in FEATURE_NAMES:
#     inputs[f] = st.number_input(f, value=0.0)

# if st.button("Predict"):
#     df = pd.DataFrame([inputs], columns=FEATURE_NAMES)
#     pred_class = int(pipeline.predict(df)[0])
#     pred_prob = float(pipeline.predict_proba(df)[0,1])
#     st.write(f"Predicted class: {pred_class}")
#     st.write(f"Probability of diabetes: {pred_prob:.2%}")
# app.py — improved Streamlit UI for Diabetes Prediction
import os
import joblib
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Diabetes Prediction", layout="wide", initial_sidebar_state="expanded")

# ---------- Helpers ----------
@st.cache_resource
def load_model(path="models/diabeties_model_rf.pkl"):
    # Try to load model from multiple candidate locations if needed
    candidates = [
        path,
        os.path.join(os.getcwd(), path),
        os.path.join("/app", path),
        "/app/models/diabeties_model_rf.pkl"
    ]
    last_exc = None
    for p in candidates:
        try:
            model = joblib.load(p)
            st.session_state["_model_path"] = p
            return model
        except FileNotFoundError as e:
            last_exc = e
            continue
        except Exception as e:
            st.error(f"Error loading model from {p}: {e}")
            raise
    raise FileNotFoundError(f"Model file not found. Tried: {', '.join(candidates)}") from last_exc

def make_input_row(label, min_val, max_val, step, default, format_str="%.2f"):
    return st.number_input(label, min_value=min_val, max_value=max_val, value=default, step=step, format=format_str)

def pretty_pct(x):
    return f"{x:.1%}"

# ---------- Load model ----------
try:
    MODEL_PATH = os.getenv("MODEL_PATH", "models/diabeties_model_rf.pkl")
    pipeline = load_model(MODEL_PATH)
except Exception as e:
    st.error("Could not load model — check model path and logs.")
    st.exception(e)
    st.stop()

# ---------- UI layout ----------
st.title("🩺 Diabetes Prediction")
st.write("Enter patient data (same features used in training). This app returns a probability and a predicted class (0 = No Diabetes, 1 = Diabetes).")

# Sidebar: presets, threshold, model info
with st.sidebar:
    st.header("Settings & Info")
    threshold = st.slider("Decision threshold (probability)", 0.0, 1.0, 0.5, 0.01,
                          help="If predicted probability >= threshold → classify as positive (diabetes). Lowering threshold increases sensitivity.")
    st.write("---")
    st.subheader("Quick presets")
    preset = st.selectbox("Choose example patient", ["Select...", "Healthy-like", "Moderate risk", "High risk"])
    if st.button("Apply preset"):
        if preset == "Healthy-like":
            st.session_state.update({
                "Pregnancies": 0, "Glucose": 85.0, "BloodPressure": 66.0, "SkinThickness": 20.0,
                "Insulin": 79.0, "BMI": 24.0, "DiabetesPedigreeFunction": 0.2, "Age": 30
            })
        elif preset == "Moderate risk":
            st.session_state.update({
                "Pregnancies": 2, "Glucose": 130.0, "BloodPressure": 80.0, "SkinThickness": 25.0,
                "Insulin": 100.0, "BMI": 30.0, "DiabetesPedigreeFunction": 0.6, "Age": 45
            })
        elif preset == "High risk":
            st.session_state.update({
                "Pregnancies": 6, "Glucose": 200.0, "BloodPressure": 95.0, "SkinThickness": 35.0,
                "Insulin": 200.0, "BMI": 37.0, "DiabetesPedigreeFunction": 1.2, "Age": 55
            })
    st.write("---")
    st.markdown("**Model loaded from:**")
    st.code(st.session_state.get("_model_path", MODEL_PATH))
    # show model info if available
    try:
        model_obj = pipeline
        # if pipeline, try to get the inner estimator
        if hasattr(pipeline, "named_steps") and "model" in pipeline.named_steps:
            model_obj = pipeline.named_steps["model"]
        st.write("Model type:", type(model_obj).__name__)
    except Exception:
        pass
    st.write("---")
    st.markdown("**Notes:**\n- This is a demo model. For clinical use, confirm with lab tests (HbA1c, fasting glucose) and consult a doctor.\n- Keep feature order same as training.")

# ---------- Main input form (two-column layout) ----------
FEATURE_NAMES = [
    "Pregnancies","Glucose","BloodPressure","SkinThickness",
    "Insulin","BMI","DiabetesPedigreeFunction","Age"
]

# Use session_state keys as inputs so presets can set them
cols = st.columns(2)
with cols[0]:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=st.session_state.get("Pregnancies", 0))
    glucose = st.number_input("Glucose (mg/dL)", min_value=0.0, max_value=500.0, value=st.session_state.get("Glucose", 85.0), step=1.0)
    bloodp = st.number_input("Blood Pressure (mm Hg)", min_value=0.0, max_value=200.0, value=st.session_state.get("BloodPressure", 66.0), step=1.0)
    skin = st.number_input("Skin Thickness (mm)", min_value=0.0, max_value=100.0, value=st.session_state.get("SkinThickness", 20.0), step=1.0)
with cols[1]:
    insulin = st.number_input("Insulin (mu U/ml)", min_value=0.0, max_value=1000.0, value=st.session_state.get("Insulin", 79.0), step=1.0)
    bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=st.session_state.get("BMI", 24.0), step=0.1, format="%.1f")
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=10.0, value=st.session_state.get("DiabetesPedigreeFunction", 0.2), step=0.01, format="%.3f")
    age = st.number_input("Age", min_value=0, max_value=120, value=st.session_state.get("Age", 30))

# Put action in center
st.write("")
predict_col = st.columns([1, 2, 1])
with predict_col[1]:
    if st.button("🔍 Predict"):
        X = pd.DataFrame([[pregnancies, glucose, bloodp, skin, insulin, bmi, dpf, age]], columns=FEATURE_NAMES)
        # model may be pipeline
        try:
            proba = pipeline.predict_proba(X)[0,1]
            base_pred = int(proba >= threshold)
        except Exception as e:
            st.error(f"Prediction error: {e}")
            raise

        # Result card
        st.markdown("### Result")
        left, right = st.columns([1, 2])
        with left:
            st.metric("Predicted class", f"{base_pred} ({'Diabetes' if base_pred==1 else 'No Diabetes'})")
            st.metric("Probability", f"{proba:.2%}")
        with right:
            # Probability bar (visual)
            st.write("#### Probability")
            st.progress(min(max(proba, 0.0), 1.0))
            # Advice
            if proba >= threshold:
                st.error("⚠️ High risk: Model suggests a likely diabetes case. Recommend confirmatory lab tests (HbA1c, fasting glucose) and consult a doctor.")
            elif proba >= 0.2:
                st.warning("Moderate risk: Monitor lifestyle and consider follow-up testing.")
            else:
                st.success("Low risk: Maintain healthy lifestyle and regular check-ups.")

        # Optionally show feature importances if available
        try:
            estimator = pipeline
            if hasattr(pipeline, "named_steps") and "model" in pipeline.named_steps:
                estimator = pipeline.named_steps["model"]
            if hasattr(estimator, "feature_importances_"):
                st.write("#### Feature importances")
                fi = estimator.feature_importances_
                fi_df = pd.DataFrame({
                    "feature": FEATURE_NAMES,
                    "importance": fi
                }).sort_values("importance", ascending=False)
                st.table(fi_df)
        except Exception:
            pass

# Footer: show sample input / export button
st.write("---")
with st.expander("Advanced: sample CSV download / batch prediction"):
    st.markdown("You can prepare a CSV file with these column names in the same order:")
    st.code(", ".join(FEATURE_NAMES))
    st.markdown("Upload CSV to run batch predictions (example not implemented in this demo).")

# Small styling tweak (optional)
st.markdown(
    """
    <style>
      .stButton>button { height:48px; font-size:16px; }
      .stProgress>div>div>div>div { background: linear-gradient(90deg, #FF0000, #FFFF00); }
    </style>
    """,
    unsafe_allow_html=True,
)
