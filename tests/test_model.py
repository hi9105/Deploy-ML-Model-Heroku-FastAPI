import sys
sys.path.append('/home/hihi1/Deploy-ML-Model-Heroku-FastAPI')

import os
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from starter_code.ml.data import process_data


def test_data_size(data: pd.DataFrame, expected_columns: list):
    """
    Test the number of rows and name of columns in data.
    """

    assert data.shape[0] > 10000
    assert data.shape[1] == len(expected_columns)
    assert list(expected_columns) == list(data.columns.values)


def test_process_data(data, categorical_features):
    _, _, encoder, lb = process_data(data, categorical_features,
                                     label=None, training=True,
                                     encoder=None, lb=None)

    assert isinstance(encoder, OneHotEncoder)
    assert isinstance(lb, LabelBinarizer)


def test_saved_model_files():
    """
    It tests the saved model, encoder and lb.
    """
    assert os.path.isfile(os.path.join("starter_code", "model_files", "random_forest_model.pkl"))
    assert os.path.isfile(os.path.join("starter_code", "model_files", "encoder.pkl"))
    assert os.path.isfile(os.path.join("starter_code", "model_files", "lb.pkl"))


def test_predict_sample(model, process_test_data):
    """
    Test model on test data - prediction expected is 0
    """
    assert model.predict(process_test_data)[0] == 0
