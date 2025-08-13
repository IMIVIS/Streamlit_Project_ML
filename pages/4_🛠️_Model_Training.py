import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from utils.data_loader import load_dataset
from utils.model_trainer import get_model_by_name, evaluate_classifier, save_model_to_file, build_pipeline_with_scaler

st.set_page_config(page_title="Model Training")

st.title("🛠️ Model Training")

dataset_key = st.selectbox("Select dataset to train on", ["iris","wine","breast_cancer","digits"])

df = load_dataset(dataset_key)
if df is None:
    st.error("Dataset could not be loaded")
else:
    st.write("### Preview")
    st.dataframe(df.head())

    # target selection
    target_col = st.selectbox("Select target/label column", df.columns, index=len(df.columns)-1)
    features = df.drop(columns=[target_col])
    target = df[target_col]

    test_size = st.slider("Test set size (%)", 10, 50, 20)
    random_state = st.number_input("Random seed", value=42, step=1)

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size/100, random_state=int(random_state))

    model_names = ["Random Forest", "SVM", "Logistic Regression", "KNN", "Decision Tree", "Gradient Boosting"]
    selected_models = st.multiselect("Select models to train/compare", model_names, default=["Random Forest", "Logistic Regression"])

    use_grid = st.checkbox("Use GridSearchCV for hyperparameter tuning (may be slow)", value=False)

    results = []
    for mname in selected_models:
        base_clf = get_model_by_name(mname)
        pipeline = build_pipeline_with_scaler(base_clf)

        if use_grid:
            # simple grid templates
            param_grid = {}
            if 'random' in mname.lower():
                param_grid = {'clf__n_estimators': [50, 100], 'clf__max_depth': [None, 5, 10]}
            if 'svm' in mname.lower():
                param_grid = {'clf__C': [0.1, 1, 10], 'clf__kernel': ['linear','rbf']}
            if 'logistic' in mname.lower():
                param_grid = {'clf__C': [0.01, 0.1, 1]}
            if 'knn' in mname.lower():
                param_grid = {'clf__n_neighbors': [3,5,7]}
            if 'decision' in mname.lower():
                param_grid = {'clf__max_depth': [None, 5, 10]}
            if 'gradient' in mname.lower():
                param_grid = {'clf__n_estimators': [50,100], 'clf__learning_rate': [0.01,0.1]}

            if param_grid:
                grid = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=1)
                with st.spinner(f"Running GridSearch for {mname}..."):
                    grid.fit(X_train, y_train)
                best = grid.best_estimator_
                res = evaluate_classifier(best, X_test, y_test)
                results.append((mname + " (GridSearch)", res, best))
            else:
                pipeline.fit(X_train, y_train)
                res = evaluate_classifier(pipeline, X_test, y_test)
                results.append((mname, res, pipeline))
        else:
            with st.spinner(f"Training {mname}..."):
                pipeline.fit(X_train, y_train)
                res = evaluate_classifier(pipeline, X_test, y_test)
                results.append((mname, res, pipeline))

    st.write("### Model Comparison")
    summary = pd.DataFrame([{"Model": r[0], **r[1]["scores"]} for r in results])
    st.dataframe(summary.set_index('Model'))

    for mname, res, clf in results:
        st.write(f"---{mname}")
        st.write(f"**Accuracy:** {res['scores']['accuracy']:.3f}")
        st.write("**Confusion Matrix**")
        st.write(res['confusion_matrix'])
        st.write("**Classification Report**")
        st.code(res['class_report'])

        # If binary classification, show ROC
        if res.get('roc') is not None:
            fpr, tpr, roc_auc = res['roc']
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
            ax.plot([0,1],[0,1],'--')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'ROC Curve - {mname}')
            ax.legend()
            st.pyplot(fig)

        # Option to download model
        if st.button(f"Save {mname} as .pkl"):
            fname = save_model_to_file(clf, f"{mname.replace(' ','_')}.pkl")
            st.success(f"Saved model to {fname}")