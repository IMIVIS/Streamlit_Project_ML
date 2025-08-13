import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from utils.data_loader import load_dataset

st.set_page_config(page_title="Visualization")

st.title("📈 Visualization")

dataset_key = st.selectbox("Select dataset for visualization", ["iris","wine","breast_cancer","digits","titanic","tips"])
df = load_dataset(dataset_key)
if df is None:
    st.error("No data loaded")
else:
    st.write("### Dataset head")
    st.dataframe(df.head())

    plot_type = st.selectbox("Choose plot type", ["Scatter", "Histogram", "Boxplot", "Pairplot", "Correlation Matrix"])

    numeric_cols = list(df.select_dtypes(include=['number']).columns)

    if plot_type == "Scatter":
        if len(numeric_cols) < 2:
            st.warning("Not enough numeric columns for scatter plot")
        else:
            x = st.selectbox("X-axis", numeric_cols, index=0)
            y = st.selectbox("Y-axis", numeric_cols, index=1)
            hue_candidates = [c for c in df.columns if df[c].nunique() < 20]
            hue = st.selectbox("Color by (optional)", [None]+hue_candidates)
            fig, ax = plt.subplots()
            if hue:
                sns.scatterplot(data=df, x=x, y=y, hue=hue, ax=ax)
            else:
                sns.scatterplot(data=df, x=x, y=y, ax=ax)
            st.pyplot(fig)

    elif plot_type == "Histogram":
        col = st.selectbox("Select numeric column", numeric_cols)
        bins = st.slider("Bins", 5, 100, 20)
        fig, ax = plt.subplots()
        sns.histplot(df[col].dropna(), bins=bins, kde=True, ax=ax)
        st.pyplot(fig)

    elif plot_type == "Boxplot":
        if not numeric_cols:
            st.warning("No numeric columns")
        else:
            col = st.selectbox("Select numeric column", numeric_cols)
            fig, ax = plt.subplots()
            sns.boxplot(y=df[col], ax=ax)
            st.pyplot(fig)

    elif plot_type == "Pairplot":
        sel = st.multiselect("Select numeric columns for pairplot (3-6 recommended)", numeric_cols, default=numeric_cols[:4])
        if len(sel) < 2:
            st.warning("Select at least 2 columns")
        else:
            st.write("Generating pairplot — this may take a few seconds...")
            fig = sns.pairplot(df[sel].dropna())
            st.pyplot(fig.fig)

    elif plot_type == "Correlation Matrix":
        fig, ax = plt.subplots()
        sns.heatmap(df[numeric_cols].corr(), annot=True, fmt='.2f', ax=ax)
        st.pyplot(fig)