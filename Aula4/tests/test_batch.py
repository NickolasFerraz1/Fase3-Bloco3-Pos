"""Testes do batch_predict."""
from pathlib import Path

import pandas as pd
import pytest

from batch_predict import predict_batch


@pytest.fixture
def csv_entrada(tmp_path):
    """Cria CSV de entrada temporário."""
    df = pd.DataFrame({
        "sepal_length": [5.1, 6.7, 6.3],
        "sepal_width": [3.5, 3.1, 3.3],
        "petal_length": [1.4, 4.4, 6.0],
        "petal_width": [0.2, 1.4, 2.5],
    })
    path = tmp_path / "input.csv"
    df.to_csv(path, index=False)
    return path


def test_batch_gera_csv_de_saida(csv_entrada, tmp_path):
    output = tmp_path / "out.csv"
    predict_batch(str(csv_entrada), str(output), "models/iris_model.pkl")
    assert output.exists()


def test_batch_adiciona_colunas_de_predicao(csv_entrada, tmp_path):
    output = tmp_path / "out.csv"
    df = predict_batch(str(csv_entrada), str(output), "models/iris_model.pkl")
    for col in ("prediction", "class_name", "confidence"):
        assert col in df.columns


def test_batch_3_classes_diferentes(csv_entrada, tmp_path):
    output = tmp_path / "out.csv"
    df = predict_batch(str(csv_entrada), str(output), "models/iris_model.pkl")
    nomes = set(df["class_name"])
    assert nomes == {"setosa", "versicolor", "virginica"}
