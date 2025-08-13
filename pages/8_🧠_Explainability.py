import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import shap
from utils.data_loader import load_dataset
from utils.model_trainer import get_model_by_name, build_pipeline_with_scaler

st.set_page_config(page_title="Explainability")

st.title("🧠 Model Explainability (SHAP)")

dataset_key = st.selectbox("Select dataset for explainability demo", ["iris","wine","breast_cancer"]) 
df = load_dataset(dataset_key)
if df is None:
    st.error("Dataset could not be loaded")
else:
    st.write(df.head())
    target_col = st.selectbox("Select target column", df.columns, index=len(df.columns)-1)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    model_name = st.selectbox("Choose model for explainability", ["Random Forest","Logistic Regression"]) 
    clf = get_model_by_name(model_name)
    pipeline = build_pipeline_with_scaler(clf)
    with st.spinner("Training model for SHAP explainer..."):
        pipeline.fit(X, y)

    # SHAP works best with tree models for TreeExplainer; fallback to KernelExplainer
    try:
        explainer = shap.Explainer(pipeline.named_steps['clf'], X)
        shap_values = explainer(X)
        st.write("### SHAP summary plot")
        fig = shap.plots.beeswarm(shap_values, show=False)
        st.pyplot(bbox_inches='tight')
    except Exception as e:
        st.write("SHAP plotting failed, trying KernelExplainer...")
        try:
            explainer = shap.KernelExplainer(pipeline.predict_proba, shap.sample(X, 50))
            shap_values = explainer.shap_values(shap.sample(X, 50))
            st.write("Computed SHAP values (KernelExplainer)")
        except Exception as e2:
            st.error(f"SHAP failed: {e2}")

    # Feature importance
    if hasattr(pipeline.named_steps['clf'], 'feature_importances_'):
        st.write("## Feature Importances")
        importances = pipeline.named_steps['clf'].feature_importances_
        feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)
        st.bar_chart(feat_imp)