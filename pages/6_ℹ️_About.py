import streamlit as st

st.set_page_config(page_title="About Project")

st.title("ℹ️ About This Project")
st.write("""

**Project Title:** ModelHub Live – Interactive ML model playground

**Objective:** Build an interactive dashboard to explore datasets, visualize features, and train machine learning models without writing code each time. The dashboard is intended as a learning and demo tool for ML concepts. Additional feature are chatbot integration, live data ingestion, and explainability tools.

**Features:**
- Multiple built-in datasets (Iris, Wine, Breast Cancer, Digits, Titanic, Tips)
- Data preview, statistics, correlation matrix
- Scatter, histogram, boxplot, pairplot visualizations
- Train & compare multiple ML models with hyperparameter choices
- Preprocessing: imputation, encoding, scaling
- Save trained models for later use
- Advanced features: Pipelines, GridSearchCV hyperparameter tuning, SHAP explainability, live data ingestion via yfinance
- Custom dataset upload and processing
- Chatbot integration for dataset queries (The performance of the bot might vary based on the AI model used and the dataset complexity.)

**How to run:**
1. Create project folder and copy files into the structure.
2. Install requirements from requirements.txt
3. Run: `python -m streamlit run 🏠Home.py`

""")