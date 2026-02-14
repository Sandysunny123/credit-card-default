import streamlit as st
import pandas as pd
import joblib

import os



from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

# -----------------------------
# Paths
# -----------------------------
MODEL_DIR = "model/saved"
DATA_PATH = "data/credit-card.csv"
TARGET_COL = "default.payment.next.month"

TEST_DATA_PATH = "data/test_data.csv"


# -----------------------------
# Load scaler and models
# -----------------------------
FEATURES  = joblib.load(f"{MODEL_DIR}/features.pkl")
scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")

models = {
    "Logistic Regression": joblib.load(f"{MODEL_DIR}/logistic.pkl"),
    "Decision Tree": joblib.load(f"{MODEL_DIR}/decision_tree.pkl"),
    "KNN": joblib.load(f"{MODEL_DIR}/knn.pkl"),
    "Naive Bayes": joblib.load(f"{MODEL_DIR}/naive_bayes.pkl"),
    "Random Forest": joblib.load(f"{MODEL_DIR}/random_forest.pkl"),
    "XGBoost": joblib.load(f"{MODEL_DIR}/xgboost.pkl"),
}


SCALED_MODELS = ["Logistic Regression", "KNN", "Naive Bayes"]


def compute_metrics(y_true, y_pred, y_prob):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="Credit Card Default Prediction",
    layout="wide"
)

st.title("Credit Card Default Prediction")
st.caption(
    "This application supports both default sample data and user-uploaded test CSV files."
)

st.markdown("##  Download Test Data ")

if os.path.exists(TEST_DATA_PATH):
    with open(TEST_DATA_PATH, "rb") as f:
        st.download_button(
            label="Download test_data.csv",
            data=f,
            file_name="test_data.csv",
            mime="text/csv"
        )
else:
    st.warning("Test dataset not found. Please run train_model.py first.")



st.write("Machine Learning Assignment 2 – Classification Models Comparison")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Controls")

view_type = st.sidebar.selectbox(
    "Select View",
    ["Model-wise Analysis", "All Models Comparison"]
)

model_name = st.sidebar.selectbox(
    "Select ML Model",
    list(models.keys())
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Test CSV File",
    type=["csv"]
)

# -----------------------------
# Load test data
# -----------------------------
if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
else:
    st.info(
        "No CSV file uploaded. "
        "Using a sample test dataset from the repository for demonstration. "
        "Upload your own test CSV from the sidebar to evaluate models on custom data."
    )
    full_df = pd.read_csv(DATA_PATH)
    test_df = full_df.sample(500, random_state=42)


X_test = test_df.drop(columns=[TARGET_COL], errors="ignore")
X_test = X_test[FEATURES]
y_test = test_df[TARGET_COL] if TARGET_COL in test_df.columns else None


if view_type == "All Models Comparison":

    st.subheader(" All Models Performance Comparison")

    if y_test is None:
        st.warning("Uploaded CSV must contain the target column for comparison.")
    else:
        rows = []

        for name, model in models.items():

            if name in SCALED_MODELS:
                X_eval = scaler.transform(X_test)
            else:
                X_eval = X_test

            y_pred = model.predict(X_eval)
            # y_prob = model.predict_proba(X_eval)[:, 1]

            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_eval)[:, 1]
            else:
                y_prob = model.predict(X_eval)


            metrics = compute_metrics(y_test, y_pred, y_prob)
            metrics["Model"] = name
            rows.append(metrics)

        comparison_df = pd.DataFrame(rows).set_index("Model")
        st.dataframe(comparison_df.round(4))

# -----------------------------
# Scale where required
# -----------------------------
else:
    st.subheader(f" Model-wise Analysis – {model_name}")

    model = models[model_name]

    if model_name in SCALED_MODELS:
        X_eval = scaler.transform(X_test)
    else:
        X_eval = X_test

    y_pred = model.predict(X_eval)

    if y_test is not None:
        # y_prob = model.predict_proba(X_eval)[:, 1]

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_eval)[:, 1]
        else:
            y_prob = model.predict(X_eval)


        st.write("### Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

        st.write("### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        st.dataframe(
            pd.DataFrame(
                cm,
                columns=["Predicted No Default", "Predicted Default"],
                index=["Actual No Default", "Actual Default"]
            )
        )
    else:
        st.warning("Target column not found. Showing predictions only.")
        st.dataframe(pd.DataFrame(y_pred, columns=["Prediction"]).head())