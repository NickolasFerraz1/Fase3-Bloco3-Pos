"""Testes para data/validate.py."""
import pytest

from data.load import load_iris_df
from data.validate import detect_outliers_iqr, validate_schema


def test_schema_iris_e_valido():
    df = load_iris_df()
    assert validate_schema(df) is True


def test_schema_coluna_faltando_falha():
    df = load_iris_df().drop(columns=["target"])
    with pytest.raises(AssertionError, match="Colunas faltando"):
        validate_schema(df)


def test_schema_tipo_errado_falha():
    df = load_iris_df()
    df["target"] = df["target"].astype(str)
    with pytest.raises(AssertionError, match="Tipo errado"):
        validate_schema(df)


def test_detect_outliers_retorna_dataframe():
    df = load_iris_df()
    outliers = detect_outliers_iqr(df, "sepal_width")
    assert outliers is not None
    assert len(outliers) < len(df)  # Algum filtro foi aplicado
