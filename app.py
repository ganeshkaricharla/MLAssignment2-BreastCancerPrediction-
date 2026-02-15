import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from lib.metrics import calculate_all_metrics, get_confusion_matrix, get_classification_report
from lib.constants import TARGET_FIELD, POS_LABEL, DROP_FIELDS
from models.logistic_regression import LogisticRegressionModel
from models.decision_tree import DecisionTreeModel
from models.knn import KNNModel
from models.naive_bayes import NaiveBayesModel
from models.random_forest import RandomForestModel
from models.xg_boost import XGBoostModel

MODEL_MAP = {
    "Logistic Regression": LogisticRegressionModel,
    "Decision Tree": DecisionTreeModel,
    "K-Nearest Neighbors": KNNModel,
    "Naive Bayes": NaiveBayesModel,
    "Random Forest": RandomForestModel,
    "XGBoost": XGBoostModel
}
st.set_page_config(
    page_title="ML Assignment 2(Model Evaluator)",
    layout="wide"
    )

st.title("Breat Cancer Classification - ML Assignment2")

st.sidebar.header("Configuration")
selected_model = st.sidebar.selectbox("Select Model", list(MODEL_MAP.keys()))
uploaded_file = st.sidebar.file_uploader("Upload Test CSV", type=["csv"])

@st.cache_resource
def load_and_train(model_name):
    model_obj = MODEL_MAP[model_name]()
    model_obj.train()
    return model_obj

with st.spinner(f"Training {selected_model}..")
    model = load_and_train(selected_model)

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    if TARGET_FIELD not in test_df.columns:
        st.error(f"Uploaded CSV must contain {TARGET_FIELD}")
        st.stop()
    y_test = test_df[TARGET_FIELD]
    X_test = test_df.drop(columns=[TARGET_FIELD])
    if selected_model == 'XGBoost':
        y_pred_encoded = model.model.predict(X_test)
        y_pred_proba = model.model.predict_proba(X_test)[:,1]
        y_pred = model.label_encoder.inverse_transform(y_pred_encoded)
    else:
        y_pred = model.model.predict(X_test)
        y_pred_proba = model.model.predict_proba(X_test)[:, 1]
    st.info("Using Uploaded CSV as testdata")
else:
    y_test = model.y_test
    y_pred, y_pred_proba = model.predict()
    st.info("Using Test data from train-test split")

st.header(f"Evaluation Metrics - {selected_model}")
metrics = calculate_all_metrics(y_test, y_pred, y_pred_proba,pos_label=POS_LABEL)
metrics_df = pd.DataFrame(
    {
        "Metric": list(metrics.keys()),
        "Score": [f"{v:.4f}" for v in metrics.values()]
    }
)
st.table(metrics_df)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Confusion Matrix")
    cm = get_confusion_matrix(y_test, y_pred)
    labels = sorted(y_test.unique())
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{selected_model} - Confusion Matrix")
    ax.pyplot(fig)
with col2:
    st.subheader("Classification Report")
    report = get_classification_report(y_test, y_pred)
    st.text(report)