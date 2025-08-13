import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Custom Dataset")

st.title("📂 Custom Dataset Trainer - Logistic Regression Only")

uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("### Dataset Preview")
        st.dataframe(df.head())

        st.write(f"Dataset shape: {df.shape}")

        # Target selection
        target_col = st.selectbox("Select target column", df.columns)
        if target_col:
            X = df.drop(columns=[target_col])
            y = df[target_col]

            # Downcast numeric types to save memory
            for col in X.select_dtypes(include=['int64', 'float64']).columns:
                X[col] = pd.to_numeric(X[col], downcast='float')

            # Identify categorical columns
            cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

            # Detect high-cardinality categoricals (>100 unique values)
            high_card_cat = [col for col in cat_cols if X[col].nunique() > 100]

            st.write(f"Detected categorical columns: {cat_cols}")
            if high_card_cat:
                st.warning(f"High cardinality categorical columns (>{100} unique values): {high_card_cat}")
                st.info("These columns will be encoded using frequency encoding to reduce dimensionality.")

            # Frequency encode high-cardinality columns
            def freq_encode(col):
                freq = X[col].value_counts() / len(X)
                return X[col].map(freq).fillna(0)

            for col in high_card_cat:
                X[col] = freq_encode(col)

            # For remaining categorical columns (low cardinality), apply one-hot encoding
            low_card_cat = [col for col in cat_cols if col not in high_card_cat]
            if low_card_cat:
                X = pd.get_dummies(X, columns=low_card_cat, drop_first=True)

            # Missing value imputation: fill numeric with median, categorical with mode
            for col in X.columns:
                if X[col].dtype in ['float32', 'float64']:
                    X[col] = X[col].fillna(X[col].median())
                else:
                    X[col] = X[col].fillna(X[col].mode().iloc[0])

            # Scaling numeric columns
            numeric_cols = X.select_dtypes(include=['float32', 'float64']).columns
            if st.checkbox("Apply Standard Scaling to numeric features", value=True):
                scaler = StandardScaler()
                X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

            # Split data
            test_size = st.slider("Test set size (%)", 10, 50, 20)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

            # Model training (only Logistic Regression)
            clf = LogisticRegression(max_iter=1000, random_state=42)
            with st.spinner("Training Logistic Regression..."):
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)

            # Results display
            st.success(f"Model Accuracy: {accuracy_score(y_test, y_pred):.3f}")
            st.subheader("Confusion Matrix")
            st.write(confusion_matrix(y_test, y_pred))
            st.subheader("Classification Report")
            st.code(classification_report(y_test, y_pred))

            # Option to save model
            if st.button("Save trained model (.pkl)"):
                import pickle
                filename = "custom_logistic_model.pkl"
                with open(filename, 'wb') as f:
                    pickle.dump(clf, f)
                st.success(f"Saved model as {filename}")

    except Exception as e:
        st.error(f"Error loading or processing dataset: {e}")
else:
    st.info("Upload a CSV file to start training.")
