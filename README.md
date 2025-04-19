# 📉 Customer Churn Prediction System

An end-to-end machine learning solution for predicting customer churn with a focus on modularity, fine-tuning, and interpretability. This system enables organizations to identify at-risk customers with confidence using multiple classification models and real-time explanations.

---

## 🔍 Project Summary

This project builds a robust churn prediction pipeline using industry-standard tools and practices. It combines model experimentation, fine-tuning, evaluation, and interpretability into a single framework, deployed via an interactive Streamlit dashboard.

---

## 🚀 Core Features

- Support for multiple classification models (Logistic Regression, Random Forest, XGBoost)
- Pipeline-based preprocessing and model training
- GridSearchCV-based hyperparameter tuning
- Comprehensive evaluation metrics: Accuracy, F1, Precision, Recall, ROC-AUC
- SHAP-based global and local interpretability
- Real-time predictions via Streamlit with input flexibility (manual or file-based)

---

## 🧠 ML Modeling & Architecture

### Data Preprocessing

Data is transformed using a modular pipeline approach leveraging `ColumnTransformer` and `Pipeline`. This ensures consistent handling of:

- Categorical features (via OneHot or Ordinal Encoding)
- Numerical features (scaling and imputation)
- Data leakage prevention during model training and evaluation

### Model Selection

We implemented and compared three models:
- **Logistic Regression**: Lightweight and interpretable baseline
- **Random Forest**: Robust ensemble model, well-suited for tabular data
- **XGBoost**: High-performance gradient boosting for optimized accuracy

Each model was wrapped inside its respective pipeline for streamlined training and deployment.

### Fine-Tuning Strategy

All models were optimized using `GridSearchCV` across relevant hyperparameters:
- Logistic Regression: `C`, `penalty`
- Random Forest: `n_estimators`, `max_depth`, `min_samples_split`
- XGBoost: `learning_rate`, `max_depth`, `n_estimators`, `subsample`

Fine-tuning was conducted within the pipeline framework to ensure reproducibility and avoid leakage.

---

## 📊 Model Evaluation

Each model was evaluated using stratified cross-validation on:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

These metrics provide a comprehensive view of model performance, especially for imbalanced datasets where accuracy alone may be misleading.

---

## 🔎 Interpretability

To provide transparency and support model decisions, SHAP (SHapley Additive exPlanations) was integrated:
- **Global Interpretability**: Feature importance plots ranked by average SHAP values
- **Local Interpretability**: Force plots for individual prediction explanations
- **Dashboard Integration**: Dynamic visualization of SHAP values for real-time insights

This helps domain experts understand not only *what* the model predicts, but *why*.

---

## 💻 Streamlit Dashboard

The system includes a user-facing Streamlit interface allowing:
- Input via manual entry or batch CSV upload
- Real-time churn prediction with model selection
- Visualization of SHAP explanations
- Metrics dashboard for comparing models

The app is optimized for both technical and non-technical users, ensuring ease of access to model insights.

---

## ⚙️ Setup Instructions

```bash
# Clone repository
git clone https://github.com/yourusername/churn-prediction.git
cd churn-prediction

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate       # or use .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Launch the Streamlit app
streamlit run app/app.py
