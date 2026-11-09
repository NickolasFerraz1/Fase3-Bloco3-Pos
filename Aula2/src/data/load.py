"""Carregamento do dataset Iris como DataFrame."""
import logging

import pandas as pd
from sklearn.datasets import load_iris

logger = logging.getLogger(__name__)


def load_iris_df() -> pd.DataFrame:
    """Carrega Iris e retorna DataFrame com colunas renomeadas."""
    iris = load_iris(as_frame=True)
    df = iris.frame
    df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "target"]
    logger.info(f"Iris carregado: {df.shape[0]} linhas x {df.shape[1]} colunas")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = load_iris_df()
    print(df.head())
    print(f"\nShape: {df.shape}")
