# app/app.py

import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Load trained model
model = joblib.load("../models/best_model.pkl")

sample_input =pd.read_csv("../data/processed_data.csv").drop_duplicates().iloc[:1]

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
st.title("📊 Customer Churn Prediction App")

st.write("""
This app uses a machine learning model trained on customer data to predict the likelihood of churn.
You can either manually input customer features or upload a CSV file.
""")

# Input Mode
mode = st.radio("Select Input Mode", ["Manual Input", "Upload CSV"])
import io

from PIL import Image
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

def predict_and_display(df):
    st.subheader("Prediction Results")

    # Drop 'Churn' column if it exists
    if 'Churn' in df.columns:
        df = df.drop(columns=['Churn'])

    preds = model.predict(df)
    probs = model.predict_proba(df)[:, 1]

    df_result = df.copy()
    df_result["Churn Probability"] = probs
    df_result["Prediction"] = preds
    df_result["Prediction"] = df_result["Prediction"].map({0: "No", 1: "Yes"})

    st.dataframe(df_result)

    # SHAP explanation
    explainer = shap.Explainer(model, df)
    shap_values = explainer(df, check_additivity=False)

    st.subheader("🔍 Model Explanation (first row only)")

    fig, ax = plt.subplots(figsize=(8, 6))
    shap.plots.waterfall(shap_values[0], max_display=8, show=False)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)


    # Save to in-memory buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    image = Image.open(buf)
    st.image(image)
    plt.close(fig)



if mode == "Manual Input":
    user_input = {}
    st.sidebar.header("📥 Enter Customer Info")
    for col in sample_input.columns:
        dtype = sample_input[col].dtype
        if dtype == object:
            user_input[col] = st.sidebar.selectbox(col, options=["Yes", "No"])  # Modify based on actual categories
        else:
            user_input[col] = st.sidebar.number_input(col, value=float(sample_input[col].values[0]))

    input_df = pd.DataFrame([user_input])
    predict_and_display(input_df)

else:
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        st.write("Preview:")
        st.dataframe(input_df.head())
        predict_and_display(input_df)
