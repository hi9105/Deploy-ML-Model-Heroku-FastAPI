"""
Author: Hiral
Date: May 2022
This  script do POST on live API deployed at Heroku with FastAPI.
"""

import json
import requests
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

url = ""

headers = {'content-type': 'application/json'}

request_data = {
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
    "native-country": "United-States"
}

if __name__ == '__main__':
    response = requests.post(url, data=json.dumps(request_data), headers=headers)

    logging.info(f"Response status code : {response.status_code}")
    logging.info(f"Response body : {response.json()}")
    logging.info("Predicted salary for the given data is : %s", response.json()['salary_prediction'])
