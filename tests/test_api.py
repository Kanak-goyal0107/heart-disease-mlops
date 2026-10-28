
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Heart Disease Prediction API"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_predict():
    payload = {
        "Age": 55,
        "Sex": "M",
        "ChestPainType": "ATA",
        "RestingBP": 140,
        "Cholesterol": 250,
        "FastingBS": 0,
        "RestingECG": "Normal",
        "MaxHR": 150,
        "ExerciseAngina": "N",
        "Oldpeak": 1.0,
        "ST_Slope": "Up"
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    result = response.json()
    assert result["prediction"] in [0, 1]
    assert 0 <= result["probability_class_0"] <= 1
    assert 0 <= result["probability_class_1"] <= 1


def test_predict_invalid_input():
    response = client.post("/predict", json={"Age": 55})
    assert response.status_code == 422
