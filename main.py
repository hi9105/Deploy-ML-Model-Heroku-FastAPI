"""
Author: Hiral
Date: May 2021
This RESTful API is using FastAPI for running Random forest classifier on Heroku
"""

import sys

sys.path.append('../Deploy-ML-Model-Heroku-FastAPI')

import pandas as pd
import os, joblib, json
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder
import logging
from starter_code.ml.data import process_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger()

cat_features = ["workclass", "education", "marital-status", "occupation", "relationship",
                "race", "sex", "native-country"]

# this is required for running DVC on Heroku
if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    # os.system("dvc remote add -d s3remote s3://modelawsbucket/amazons3folder")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

app = FastAPI()


# Alias generator function to match data source field names.
def underscore_to_hyphen(string: str) -> str:
    return string.replace('_', '-')


# Declare the data object with its components and their type.
class CensusData(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    class Config:
        schema_extra = {
            "example": {
                "age": 22,
                "workclass": "Private",
                "fnlgt": 201490,
                "education": "HS-grad",
                "education-num": 9,
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Own-child",
                "race": "White",
                "sex": "Male",
                "capital-gain": 0,
                "capital-loss": 0,
                "hours-per-week": 20,
                "native-country": "United-States",
            }
        }
        alias_generator = underscore_to_hyphen


model = joblib.load(os.path.join("starter_code", "model_files", "random_forest_model.pkl"))
encoder = joblib.load(os.path.join("starter_code", "model_files", "encoder.pkl"))
lb = joblib.load(os.path.join("starter_code", "model_files", "lb.pkl"))


# run before the application starts.
@app.on_event("startup")
async def startup_event():
    logger.info("Additionally model, encoder and lb can be loaded on startup for faster predictions")
    # global model, encoder, lb
    # model = joblib.load(os.path.join("starter_code", "model_files", "random_forest_model.pkl"))
    # encoder = joblib.load(os.path.join("starter_code", "model_files", "encoder.pkl"))
    # lb = joblib.load(os.path.join("starter_code", "model_files", "lb.pkl"))


# GET on the root to give a welcome message.
@app.get("/")
async def welcome_message():
    """
    GET on the root giving a welcome message.
    """

    return {
        "message": "Welcome message : This is FastAPI web framework to predict whether a person has salary > 50K or "
                   "<= 50K !!!"}


# POST to do model inference.
# FastAPI webui = http://127.0.0.1:8000/docs#/default/post_model_inference_predict_post
@app.post("/predict")
async def post_model_inference(postData: CensusData):
    """
    POST that does model inference.
    Output:
        0 : salary <=50K
        1 : salary >50K
    """

    # post_data_df = pd.DataFrame.from_dict([postData.dict(by_alias=True)])
    post_data_df = pd.DataFrame(jsonable_encoder(postData), index=[0])
    logger.info(f"Data used for making a post request is: {post_data_df}")

    logger.info("Processing post data...")
    X, _, _, _ = process_data(
        post_data_df, categorical_features=cat_features, training=False, encoder=encoder, lb=lb)

    logger.info("Predicting post data...")
    preds = model.predict(X)[0]
    logger.info("Predicted salary for the given data is %s", preds)
    result = {"salary_prediction": "<=50K" if preds == 0 else ">50K"}
    return result


#  run when the application is shutting down.
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutdown")
