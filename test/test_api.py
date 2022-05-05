from fastapi.testclient import TestClient
from main import app

# Instantiate the testing client with our app.
client = TestClient(app)


def test_get_on_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"message": "Welcome message : This is FastAPI web framework to predict whether a person has "
                                   "salary > 50K or <= 50K !!!"}


def test_post_model_inference_less50K(test_data_less50K):
    r = client.post("/predict", json=test_data_less50K)
    assert r.status_code == 200
    assert r.json()['salary_prediction'] == 0


def test_post_model_inference_more50K(test_data_more50K):
    r = client.post("/predict", json=test_data_more50K)
    assert r.status_code == 200
    assert r.json()['salary_prediction'] == 1
