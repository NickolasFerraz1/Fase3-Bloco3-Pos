"""Testes da API Flask."""
import pytest

from api.app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_home_retorna_200(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "message" in r.get_json()


def test_health_retorna_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.get_json() == {"status": "ok"}


def test_predict_setosa(client):
    r = client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2]})
    assert r.status_code == 200
    data = r.get_json()
    assert data["prediction"] == 0
    assert data["class_name"] == "setosa"
    assert 0 <= data["confidence"] <= 1


def test_predict_versicolor(client):
    r = client.post("/predict", json={"features": [6.7, 3.1, 4.4, 1.4]})
    assert r.status_code == 200
    assert r.get_json()["class_name"] == "versicolor"


def test_predict_virginica(client):
    r = client.post("/predict", json={"features": [6.3, 3.3, 6.0, 2.5]})
    assert r.status_code == 200
    assert r.get_json()["class_name"] == "virginica"


def test_predict_sem_features_retorna_400(client):
    r = client.post("/predict", json={"x": [1, 2, 3, 4]})
    assert r.status_code == 400


def test_predict_lista_tamanho_errado_retorna_400(client):
    r = client.post("/predict", json={"features": [1, 2]})
    assert r.status_code == 400


def test_predict_features_nao_e_lista_retorna_400(client):
    r = client.post("/predict", json={"features": "abc"})
    assert r.status_code == 400
