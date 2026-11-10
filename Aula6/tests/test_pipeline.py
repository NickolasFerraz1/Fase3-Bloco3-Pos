"""Testes do pipeline (Vídeo 6.1).

Cada função é testável isoladamente - o ganho concreto de quebrar o código monolítico.
"""
import pandas as pd
import pytest

from src.ml_pipeline.data import clean_data, load_data
from src.ml_pipeline.model import MODELS, build_model
from src.ml_pipeline.pipeline import train


def test_load_data_carrega_iris():
    # csv inexistente força o fallback para o scikit-learn
    df = load_data(csv_path="nao/existe.csv")
    assert len(df) == 150
    assert "target" in df.columns


def test_clean_data_remove_duplicatas():
    df = pd.DataFrame({"a": [1, 1, 2], "b": [1.0, 1.0, 2.0]})
    out = clean_data(df)
    assert len(out) == 2


def test_build_model_desconhecido_levanta_erro():
    with pytest.raises(ValueError):
        build_model("inexistente")


def test_modelos_disponiveis():
    assert {"rf", "rf_small", "logreg"}.issubset(set(MODELS))


def test_train_atinge_boa_accuracy():
    result = train(model_name="rf", seed=42)
    assert result["accuracy"] > 0.8
    assert result["n_rows"] > 0
