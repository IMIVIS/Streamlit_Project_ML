import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
import seaborn as sns

AVAILABLE_DATASETS = {
    'iris': {
        'name': 'Iris',
        'short': 'Classic 3-class iris dataset',
        'long': 'Iris dataset by Fisher (1936). 150 samples, 4 features, 3 classes.',
        'source': 'sklearn.datasets'
    },
    'wine': {
        'name': 'Wine',
        'short': 'Wine recognition dataset',
        'long': 'Wine dataset from UCI. Chemical analysis of wines grown in the same region in Italy.',
        'source': 'sklearn.datasets'
    },
    'breast_cancer': {
        'name': 'Breast Cancer',
        'short': 'Breast cancer wisconsin',
        'long': 'Breast cancer (diagnostic) dataset for classification (malignant/benign).',
        'source': 'sklearn.datasets'
    },
    'digits': {
        'name': 'Digits',
        'short': 'Handwritten digit recognition',
        'long': '8x8 images of handwritten digits; used for multiclass classification demos.',
        'source': 'sklearn.datasets'
    },
    'titanic': {
        'name': 'Titanic (sample)',
        'short': 'Passenger survival dataset (sample from seaborn)',
        'long': 'Titanic dataset containing passenger info and survival labels — useful for preprocessing demos.',
        'source': 'seaborn'
    },
    'tips': {
        'name': 'Tips (sample)',
        'short': 'Restaurant tips dataset (seaborn)',
        'long': 'Tips dataset — useful for regression or classification after transformation.',
        'source': 'seaborn'
    }
}


def load_dataset(key):
    key = key.lower()
    try:
        if key == 'iris':
            data = load_iris(as_frame=True).frame
            data['target'] = load_iris(as_frame=True).target
            return data
        if key == 'wine':
            data = load_wine(as_frame=True).frame
            data['target'] = load_wine(as_frame=True).target
            return data
        if key == 'breast_cancer':
            data = load_breast_cancer(as_frame=True).frame
            data['target'] = load_breast_cancer(as_frame=True).target
            return data
        if key == 'digits':
            data = load_digits(as_frame=True).frame
            data['target'] = load_digits(as_frame=True).target
            return data
        if key == 'titanic':
            return sns.load_dataset('titanic')
        if key == 'tips':
            return sns.load_dataset('tips')
    except Exception as e:
        print('load_dataset error', e)
        return None
    return None