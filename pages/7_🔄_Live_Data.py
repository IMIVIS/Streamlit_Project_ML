import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from utils.model_trainer import get_model_by_name, evaluate_classifier, save_model_to_file, build_pipeline_with_scaler

st.set_page_config(page_title="Live Data")

st.title("🔄 Live Data - Quick Fetch & Model")

symbol = st.text_input("Enter stock ticker (e.g., AAPL, MSFT)")
notice = st.text("Look for yfinance ticker symbols on internet or Yahoo Finance.")
period = st.selectbox("Period", ["1mo","3mo","6mo","1y"], index=3)

if st.button("Fetch Data") and symbol:
    end = datetime.today()
    start = end - timedelta(days=365)
    try:
        df = yf.download(symbol, period=period)
        if df.empty:
            st.error("No data fetched. Check the ticker symbol.")
        else:
            st.write(df.tail())
            st.line_chart(df['Close'])
            # quick feature engineering for demo: use past returns
            df['Return'] = df['Close'].pct_change()
            df['TargetUp'] = (df['Return'].shift(-1) > 0).astype(int)
            df = df.dropna()
            st.write("### Prepared dataset (for demo classification: next-day up/down)")
            st.dataframe(df[['Close','Return','TargetUp']].head())

            X = df[['Return']]
            y = df['TargetUp']
            split = int(len(X)*0.8)
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]

            clf = get_model_by_name('Random Forest')
            pipeline = build_pipeline_with_scaler(clf)
            pipeline.fit(X_train, y_train)
            res = evaluate_classifier(pipeline, X_test, y_test)
            st.write(f"Accuracy on recent data: {res['scores']['accuracy']:.3f}")
            if st.button("Save model for this ticker"):
                fname = save_model_to_file(pipeline, f"{symbol}_model.pkl")
                st.success(f"Saved model to {fname}")
    except Exception as e:
        st.error(f"Failed to fetch data: {e}")
