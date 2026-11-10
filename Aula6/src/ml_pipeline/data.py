"""Carga e limpeza de dados (Vídeo 6.1 - SRP).

SRP (Single Responsibility): cada função faz UMA coisa.
- load_data: traz o dataset.
- clean_data: limpa o dataset.
Nada aqui treina, avalia ou imprime relatório.
"""
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET = "target"


def load_data(csv_path: str = "data/iris.csv") -> pd.DataFrame:
    """Carrega o Iris de um CSV (se existir) ou do scikit-learn."""
    if csv_path and Path(csv_path).exists():
        return pd.read_csv(csv_path)
    frame = load_iris(as_frame=True).frame  # pylint: disable=no-member
    frame.columns = FEATURES + [TARGET]
    return frame


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preenche nulos numéricos pela média e remove duplicatas."""
    out = df.copy()
    numeric = out.select_dtypes(include="number").columns
    out[numeric] = out[numeric].fillna(out[numeric].mean())
    return out.drop_duplicates().reset_index(drop=True)
