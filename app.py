import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score,
    f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

st.set_page_config(page_title="ML Assignment 2 - Credit Default", layout="wide")
st.title("Credit Card Default Prediction — Model Comparison")

# -----------------------------
# Configuration
# -----------------------------
TARGET_COL = "default.payment.next.month"

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest (Ensemble)": "model/random_forest_ensemble.pkl",
    "XGBoost (Ensemble)": "model/xgboost_ensemble.pkl",
}

SCALED_MODELS = {"Logistic Regression", "KNN", "Naive Bayes"}  # these use scaler
SCALER_PATH = "model/scaler.pkl"

# -----------------------------
# Helper: probability
# -----------------------------
def get_binary_proba(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
    else:
        raise ValueError("Model has neither predict_proba nor decision_function.")

# -----------------------------
# Sidebar controls
# -----------------------------
#### st.sidebar.header("Controls")

# uploaded_file = st.sidebar.file_uploader("Upload TEST CSV (recommended)", type=["csv"])
###
import os
import streamlit as st
import pandas as pd

DEFAULT_TEST_PATH = "model/data/test.csv"

st.sidebar.subheader("Test Data Source")

uploaded_file = st.sidebar.file_uploader("Upload TEST CSV (optional)", type=["csv"])

# Initialize state
if "use_default" not in st.session_state:
    st.session_state.use_default = False

# Button to use default
if st.sidebar.button("Use default test.csv from GitHub"):
    st.session_state.use_default = True

# If user uploads a file, prefer the upload and turn off default
if uploaded_file is not None:
    st.session_state.use_default = False
model_name = st.sidebar.selectbox("Select Model", list(MODEL_FILES.keys()))

# Decide which data to load
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Using uploaded test CSV.")
elif st.session_state.use_default:
    if not os.path.exists(DEFAULT_TEST_PATH):
        st.error(f"Default test file not found at: {DEFAULT_TEST_PATH}")
        st.stop()
    df = pd.read_csv(DEFAULT_TEST_PATH)
    st.info("Using default test.csv from repository.")
else:
    st.warning("Upload a test CSV or click 'Use default test.csv from GitHub'.")
    st.stop()

# model_name = st.sidebar.selectbox("Select Model", list(MODEL_FILES.keys()))
# Load model
model_path = MODEL_FILES[model_name]
if not os.path.exists(model_path):
    st.error(f"Model file not found: {model_path}. Make sure it exists in GitHub.")
    st.stop()

model = joblib.load(model_path)

# Load scaler if needed
scaler = None
if model_name in SCALED_MODELS:
    if not os.path.exists(SCALER_PATH):
        st.error("Scaler not found. Expected at model/scaler.pkl")
        st.stop()
    scaler = joblib.load(SCALER_PATH)

# -----------------------------
# Main logic
# -----------------------------
if uploaded_file is None:
    st.info("Upload a CSV file from the sidebar to evaluate the selected model.")
    st.stop()

df = pd.read_csv(uploaded_file)

st.subheader("Uploaded Data Preview")
st.dataframe(df.head(10), use_container_width=True)

# Drop ID column if present
if "ID" in df.columns:
    df = df.drop(columns=["ID"])

# Check target presence
has_target = TARGET_COL in df.columns

if has_target:
    X_test = df.drop(columns=[TARGET_COL])
    y_test = df[TARGET_COL]
else:
    X_test = df.copy()
    y_test = None
    st.warning(f"Target column '{TARGET_COL}' not found in upload. Metrics will not be computed.")

# Apply scaling if needed
X_in = X_test
if scaler is not None:
    X_in = scaler.transform(X_test)

# Predict
y_pred = model.predict(X_in)

# Predict probabilities for AUC
y_prob = None
try:
    y_prob = get_binary_proba(model, X_in)
except:
    pass

st.subheader(f"Selected Model: {model_name}")

# If target exists, compute metrics
if y_test is not None:
    col1, col2, col3 = st.columns(3)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)

    auc = None
    if y_prob is not None:
        try:
            auc = roc_auc_score(y_test, y_prob)
        except:
            auc = None

    with col1:
        st.metric("Accuracy", f"{accuracy:.4f}")
        st.metric("AUC", f"{auc:.4f}" if auc is not None else "N/A")

    with col2:
        st.metric("Precision", f"{precision:.4f}")
        st.metric("Recall", f"{recall:.4f}")

    with col3:
        st.metric("F1 Score", f"{f1:.4f}")
        st.metric("MCC", f"{mcc:.4f}")

    # Confusion Matrix + Classification Report
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)

    cm_df = pd.DataFrame(cm, index=["Actual: No Default", "Actual: Default"], columns=["Predicted: No Default", "Predicted: Default"])
    st.dataframe(cm_df, use_container_width=True)
    # st.text(classification_report(y_test, y_pred, digits=4, zero_division=0))
    from sklearn.metrics import classification_report
    report_dict = classification_report(y_test, y_pred, digits=4, zero_division=0, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()

    st.subheader("Classification Report")
    st.dataframe(report_df, use_container_width=True)

else:
    # If no target, show predictions only
    st.subheader("Predictions")
    out = X_test.copy()
    out["prediction"] = y_pred
    if y_prob is not None:
        out["prob_default_1"] = y_prob
    st.dataframe(out.head(20), use_container_width=True)

