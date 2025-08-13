import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils.data_loader import load_dataset

st.set_page_config(page_title="Data Overview")

st.title("📊 Data Overview")

dataset_key = st.selectbox("Select built-in dataset", ["iris","wine","breast_cancer","digits","titanic","tips"], index=0)

df = load_dataset(dataset_key)
if df is None:
    st.error("Dataset could not be loaded")
else:
    st.subheader("Preview of data")
    st.dataframe(df.head())

    st.subheader("Statistical summary (numeric columns)")
    st.dataframe(df.describe())

    st.subheader("Class / Target Distribution (if available)")
    # heuristics to find target column
    possible_targets = [c for c in df.columns if df[c].nunique()<20 and df[c].dtype != 'float']
    if possible_targets:
        tgt = st.selectbox("Select column to treat as target (optional)", [None]+possible_targets)
    else:
        tgt = None

    if tgt:
        st.bar_chart(df[tgt].value_counts())

    st.subheader("Correlation matrix")
    numeric_df = df.select_dtypes(include=["number"]).dropna(axis=1, how='all')
    if numeric_df.shape[1] >= 2:
        fig, ax = plt.subplots()
        sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', ax=ax)
        st.pyplot(fig)
    else:
        st.write("Not enough numeric columns to show correlation.")
