import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.model import predict_product

client = TestClient(app)

def test_predict_product():
    result = predict_product("Basic cotton terry shorts", "Shorts in lightweight organic cotton sweatshirt fabric")

    assert "predicted_group" in result
    assert "probabilities" in result
    assert isinstance(result["predicted_group"], str)
    assert isinstance(result["probabilities"], dict)

def test_predict_probabilities():
    result = predict_product("Basic cotton terry shorts", "Shorts in lightweight organic cotton sweatshirt fabric")
    probabilities = result["probabilities"]

    assert len(probabilities) == 5
    assert all(0 <= value <= 1 for value in probabilities.values())
    assert sum(probabilities.values()) == pytest.approx(1.0)

def test_heath():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict():
    response = client.post("/predict", json = {
        "prod_name": "Basic cotton terry shorts",
        "detail_desc": (
        "Shorts in lightweight organic cotton"
        "sweatshirt fabric"
        )})

    assert response.status_code == 200
    data = response.json()
    assert "predicted_group" in data
    assert "probabilities" in data

@pytest.mark.parametrize(
    "prod_name, detail_desc",[(
            "Basic cotton terry shorts",
            "Shorts in lightweight organic cotton sweatshirt fabric"
        ),
        (
            "Cotton jersey top",
            "Basic top in soft cotton jersey"
        ),
        (
            "Knitted cardigan",
            "Soft knitted cardigan with long sleeves"
        )]
        )

def test_predict_diff_products(prod_name, detail_desc):
    response = client.post("/predict", json = {"prod_name": prod_name, "detail_desc": detail_desc})

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["predicted_group"], str)
    assert len(data["probabilities"]) == 5

def test_invalid_input():
    response = client.post("/predict", json = {"prod_name": 67, "detail_desc": "text text text"})
    assert response.status_code == 422

def test_empty_text():
    response = client.post("/predict", json = {"prod_name": "", "detail_desc": ""})

    assert response.status_code == 400
    data = response.json()
    assert "detail" in data


