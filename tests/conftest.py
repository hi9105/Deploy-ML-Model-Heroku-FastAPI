import pytest, os, sys
import pandas as pd
import joblib

sys.path.append('../Deploy-ML-Model-Heroku-FastAPI')
#sys.path.insert(0, os.getcwd())
from starter_code.ml.data import process_data


@pytest.fixture(scope='session')
def data(request):
    #ROOT_DIR = os.path.abspath(os.curdir)
    #data = pd.read_csv(os.path.join(ROOT_DIR, 'starter_code/data/cleanCensusData.csv'))
    data = pd.read_csv("./starter_code/data/cleanCensusData.csv")
    return data


@pytest.fixture(scope="session")
def expected_columns():
    return ['age',
            'workclass',
            'fnlgt',
            'education',
            'education-num',
            'marital-status',
            'occupation',
            'relationship',
            'race',
            'sex',
            'capital-gain',
            'capital-loss',
            'hours-per-week',
            'native-country',
            'salary']


@pytest.fixture(scope='session')
def categorical_features():
    return [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country",
    ]


@pytest.fixture(scope='session')
def model(request):
    model = joblib.load(os.path.join("starter_code", "model_files", "random_forest_model.pkl"))
    return model


@pytest.fixture(scope='session')
def lb(request):
    lb = joblib.load(os.path.join("starter_code", "model_files", "lb.pkl"))
    return lb


@pytest.fixture(scope='session')
def encoder(request):
    encoder = joblib.load(os.path.join("starter_code", "model_files", "encoder.pkl"))
    return encoder


@pytest.fixture(scope='session')
def process_test_data(test_data_less50K, encoder, lb):
    """
    Test data sample processing
    """

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    df = pd.DataFrame(test_data_less50K, index=[0])

    X, _, _, _ = process_data(df, categorical_features=cat_features,
                              label=None, training=False, encoder=encoder, lb=lb)
    return X


@pytest.fixture(scope='session')
def test_data_less50K(request):
    data_less50K = {
        'age': 58,
        'workclass': 'Private',
        'fnlgt': 151910,
        'education': 'HS-grad',
        'education-num': 9,
        'marital-status': 'Widowed',
        'occupation': 'Adm-clerical',
        'relationship': 'Unmarried',
        'race': 'White',
        'sex': 'Female',
        'capital-gain': 0,
        'capital-loss': 0,
        'hours-per-week': 40,
        'native-country': 'United-States'
        #'salary': '<=50K'
    }

    return data_less50K


@pytest.fixture(scope='session')
def test_data_more50K(request):
    data_more50K = {
        'age': 40,
        'workclass': 'Private',
        'fnlgt': 154374,
        'education': 'HS-grad',
        'education-num': 9,
        'marital-status': 'Married-civ-spouse',
        'occupation': 'Machine-op-inspct',
        'relationship': 'Husband',
        'race': 'White',
        'sex': 'Male',
        'capital-gain': 0,
        'capital-loss': 0,
        'hours-per-week': 40,
        'native-country': 'United-States'
        #'salary': '>50K'
    }

    return data_more50K
