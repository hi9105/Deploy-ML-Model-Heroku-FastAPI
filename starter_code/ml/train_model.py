# Script to train machine learning model.

# necessary imports.
import yaml
from yaml.loader import SafeLoader

import pandas as pd
from sklearn.model_selection import train_test_split
from data import process_data, clean_data
from model import train_model, compute_model_metrics, inference, save_model
from slices_performance import slice_performance

with open('./params.yaml', "rb") as f:
    params = yaml.load(f, Loader=SafeLoader)

# load the data.
data = pd.read_csv(params["data"]["raw_data"])

# clean the data.
data = clean_data(data)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=params["modeling"]["test_size"])

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
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True)

# Proces the test data.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb)

# Train model.
random_forest_model = train_model(X_train, y_train)

# Calculate prediction of model.
preds = inference(random_forest_model, X_test)

# Compute results.
precision, recall, fbeta = compute_model_metrics(y_test, preds)

# save model and other files.
save_model(random_forest_model, encoder, lb)

# outputs the performance of the model on slices of the categorical features.
slice_performance(data, random_forest_model, encoder, lb, cat_features, label='salary')
