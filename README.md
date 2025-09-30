# Diabetes Prediction Project

This project is a **Diabetes Prediction System** using a **Random Forest Classifier** with **SMOTE oversampling**. The model predicts whether a patient has diabetes based on health parameters. It includes both a **training notebook** and a **web app** for real-time predictions.

---

## 📁 Project Structure

```
diabetes_project/
│
├─ notebooks/             # Jupyter notebooks for training and EDA
│    └─ train_model.ipynb
│
├─ app/                   # Web app (Streamlit / Flask)
│    ├─ app.py            # Main app file
│    └─ templates/        # (Optional, for Flask HTML templates)
│
├─ models/                # Saved trained model pipelines
│    └─ diabeties_model_rf.pkl
│
├─ data/                  # Dataset folder
│    └─ diabetes.csv
│
├─ requirements.txt       # Python dependencies
└─ README.md              # Project documentation
```

---

## 🥰 Features

* Train a **Random Forest model** on the diabetes dataset.
* Handle **class imbalance** using SMOTE.
* Save the **pipeline (Scaler + Model)** for future predictions.
* Web app for user-friendly **input & prediction**:

  * Streamlit web app for interactive predictions.
  * Flask REST API for integration with other applications.
* Support for **single and multiple patient inputs**.
* Probability-based predictions with class label.

---

## 💻 Installation

1. Clone the repository:

```bash
git clone https://github.com/mujahidul885/Diabetes-Prediction-App.git
cd diabetes_project
```

2. Create and activate a virtual environment:

```bash
# Using conda
conda create -n diabetes_env python=3.11 -y
conda activate diabetes_env

# Or using venv
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Ensure `ipykernel` is installed for notebooks:

```bash
conda install -n diabetes_env ipykernel --update-deps --force-reinstall
```

---

## 🗑️ Usage

### 1. Train Model

Run the Jupyter notebook:

```bash
jupyter lab notebooks/train_model.ipynb
```

* The trained **pipeline** will be saved as `models/diabeties_model_rf.pkl`.

### 2. Run Streamlit Web App

```bash
streamlit run app/app.py
```

* Open your browser → `http://localhost:8501`
* Enter patient health data → see prediction and probability.

### 3. Run Flask API (Optional)

```bash
python app/flask_app.py
```

* POST JSON input to `/predict` endpoint.
* Example JSON:

```json
{
  "Pregnancies": 2,
  "Glucose": 120,
  "BloodPressure": 70,
  "SkinThickness": 20,
  "Insulin": 85,
  "BMI": 25.3,
  "DiabetesPedigreeFunction": 0.351,
  "Age": 30
}
```

---

## 📊 Dataset

* Source: `data/diabetes.csv`
* Features:

  * Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
* Target:

  * Outcome: 0 (No diabetes), 1 (Diabetes)

---

## 🚀 Deployment

* Streamlit Cloud: Easy web deployment.
* Render / Railway: Host Streamlit or Flask app.
* Optional Docker container for cloud deployment.

---

## 🚀 Deployment

* Streamlit Cloud: Easy web deployment.
* Render / Railway: Host Streamlit or Flask app.
* Optional Docker container for cloud deployment.

**Live Web App:** [Click here to try the Diabetes Prediction App](https://diabetes-prediction-app-x8kj.onrender.com/)

## ⚡ Notes

* Ensure **feature order** remains the same as training when making predictions.
* The model uses **Random Forest + SMOTE** to handle class imbalance and improve recall.
* Adjust probability threshold if you want to increase sensitivity for diabetes detection.

---

## 👤 Author

**Mujahidul Islam**
Email: [mujahidulI845455@gmail.com](mailto:mujahidulI845455@gmail.com)
Phone: +91 8603629937


