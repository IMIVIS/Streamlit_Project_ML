import streamlit as st
from utils.data_loader import AVAILABLE_DATASETS

# Basic page config
st.set_page_config(page_title="AI/ML Dashboard", page_icon="🤖", layout="wide")

st.title("📌 ModelHub Live – Interactive ML model playground")
st.markdown(
    """
    **Overview**

    This multi-page Streamlit application provides:
    - Built-in datasets (Iris, Wine, Breast Cancer, Digits, Titanic, Tips)
    - Dataset exploration and visualizations
    - Model training and evaluation (on built-in and custom CSVs)
    - Preprocessing utilities (imputation, scaling, encoding)
    - Advanced features: pipelines, hyperparameter tuning, model persistence, explainability (SHAP/LIME), live data ingest

    Use the left sidebar to navigate between pages. Each page is located in the `pages/` folder so Streamlit will automatically expose them.
    """
)

st.info("Open the sidebar to select pages (Dataset Info, Data Overview, Visualization, Model Training, Custom Dataset, Live Data, Explainability, About).")

st.write("---")

st.write("Project structure: `Home.py`, `pages/`, `utils/`, `requirements.txt`.")