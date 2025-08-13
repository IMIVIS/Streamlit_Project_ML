import streamlit as st
from utils.data_loader import load_dataset, AVAILABLE_DATASETS

st.set_page_config(page_title="Dataset Info")

st.title("📚 Dataset Information")

dataset_key = st.selectbox("Select built-in dataset", list(AVAILABLE_DATASETS.keys()))
meta = AVAILABLE_DATASETS[dataset_key]

st.header(meta['name'])
st.write(meta['long'])
st.write("\n**Source / Notes:**")
st.write(meta.get('source','Not provided'))

# Show sample and feature names if available
if st.button("Load sample preview"):
    df = load_dataset(dataset_key)
    st.write(df.head())
    st.write("\n**Columns:**")
    st.write(list(df.columns))