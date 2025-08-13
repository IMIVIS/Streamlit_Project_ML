import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def get_model_by_name(name):
    name = name.lower()
    if 'random' in name:
        return RandomForestClassifier(random_state=42)
    if 'svm' in name or 'support' in name:
        return SVC(probability=True, random_state=42)
    if 'logistic' in name:
        return LogisticRegression(max_iter=2000, random_state=42)
    if 'knn' in name:
        return KNeighborsClassifier()
    if 'decision' in name:
        return DecisionTreeClassifier(random_state=42)
    if 'gradient' in name:
        return GradientBoostingClassifier(random_state=42)
    return RandomForestClassifier(random_state=42)


def build_pipeline_with_scaler(clf):
    """Wrap classifier into a pipeline with StandardScaler named steps: ('scaler','clf')"""
    return Pipeline([('scaler', StandardScaler()), ('clf', clf)])


def evaluate_classifier(clf, X_test, y_test):
    out = {}
    y_pred = clf.predict(X_test)
    scores = {}
    scores['accuracy'] = accuracy_score(y_test, y_pred)
    out['scores'] = scores
    out['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()
    out['class_report'] = classification_report(y_test, y_pred)

    # ROC/AUC for binary classification
    try:
        if len(set(y_test.tolist())) == 2:
            if hasattr(clf, 'predict_proba'):
                probs = clf.predict_proba(X_test)[:,1]
            elif hasattr(clf, 'decision_function'):
                probs = clf.decision_function(X_test)
            else:
                probs = None
            if probs is not None:
                fpr, tpr, _ = roc_curve(y_test, probs)
                roc_auc = auc(fpr, tpr)
                out['roc'] = (fpr.tolist(), tpr.tolist(), float(roc_auc))
    except Exception:
        pass

    return out


def save_model_to_file(clf, filename='model.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(clf, f)
    return filename