"""
Author: Hiral
Date: May 2022
This function outputs the performance of the model on slices of the data.
It uses saved model, encoder and lb.
It output the performance on slices of just the categorical features.
"""

from model import inference, compute_model_metrics
from data import process_data
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s- %(message)s")
logger = logging.getLogger()


def slice_performance(df, model, encoder, lb, cat_features, label='salary'):
    """
    Outputs the performance of the model on slices of the categorical features data.

    Inputs
    ------
    df : pd.DataFrame
        Dataframe containing clean data.
    model : pkl file
        Trained saved model.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer.
    cat_features : list of string.
        categorical features.
    label : str
        Name of the label column in `df`.

    Returns
    -------
    None
    """
    slice_performance_report = []
    for categorical in cat_features:
        logger.info(f"Start : Model performance on slice of categorical feature : {categorical}")
        for value in df[categorical].unique():
            df_temp = df[df[categorical] == value]

            X, y, _, _ = process_data(df_temp,
                                      categorical_features=cat_features,
                                      label=label, training=False, encoder=encoder, lb=lb)

            preds = inference(model, X)

            precision, recall, fbeta = compute_model_metrics(y, preds)

            slice_performance_report.append((categorical, value, precision, recall, fbeta))

            logger.info(f"Success : Model performance on slice of categorical feature : {categorical}")
