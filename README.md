# Telco Customer Churn Prediction (XGBoost + SHAP)

A machine learning project to predict customer churn in the telecom sector and explain why customers leave using SHAP values.

## Overview
Retaining existing customers is significantly cheaper than acquiring new ones. The goal of this project is to build an early warning system that flags at-risk customers and highlights the key drivers behind their decision.

## Pipeline
- **Dataset:** IBM Telco Customer Churn (~7,000 customers).
- **Data Prep:** Cleaned missing values in `TotalCharges`, applied one-hot encoding, and used stratified splitting.
- **Modeling:** Trained an **XGBoost Classifier** with `scale_pos_weight` to handle class imbalance without synthetic sampling.
- **Explainability:** Used **SHAP (SHapley Additive exPlanations)** to interpret both global feature importance and individual customer risk profiles.

## Key Results
- **ROC-AUC:** `0.84`
- **Top Churn Drivers:** Month-to-month contracts, payment via Electronic Check, and absence of Tech Support.
- **Business Impact:** Targeting the top at-risk cohort with proactive retention offers can save an estimated ~€25,000+ in customer lifetime value on the test sample.

## Quickstart
```bash
git clone [https://github.com/](https://github.com/)<MRigoni10>/telco-churn-explainable-ai.git
cd telco-churn-explainable-ai
pip install -r requirements.txt
