# Script to train machine learning model.

# necessary imports.
import yaml
from yaml.loader import SafeLoader
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from data import process_data, clean_data
from model import train_model, compute_model_metrics, inference, save_model
from slices_performance import slice_performance

logging.basicConfig(level=logging.INFO, format="%(asctime)s- %(message)s")
logger = logging.getLogger()

with open("../params.yaml", "rb") as f:
    params = yaml.load(f, Loader=SafeLoader)

# load the data.
logger.info("Loading raw census data...")
#data = pd.read_csv(params["data"]["raw_data"])
data = pd.read_csv("../data/rawCensusData.csv")

# clean the data.
logger.info("Cleaning data...")
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

logger.info("Processing train data...")
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True)

# Proces the test data.
logger.info("Processing test data...")
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb)

# Train model.
logger.info("Training model...")
random_forest_model = train_model(X_train, y_train)

# Calculate prediction of model.
logger.info("Calculating model predictions...")
preds = inference(random_forest_model, X_test)

# Compute results.
logger.info("Computing results...")
precision, recall, fbeta = compute_model_metrics(y_test, preds)
logger.info(f"Precision score: {precision}")
logger.info(f"Recall score: {recall}")
logger.info(f"Fbeta score : {fbeta}")

# save model and other files.
logger.info("Saving model, encoder and lb...")
save_model(random_forest_model, encoder, lb)

# outputs the performance of the model on slices of the categorical features.
logger.info("Outputs the performance of the model on slices of the categorical features...")
slice_performance(data, random_forest_model, encoder, lb, cat_features, label='salary')
logger.info("SUCCESS : Finished !!!")