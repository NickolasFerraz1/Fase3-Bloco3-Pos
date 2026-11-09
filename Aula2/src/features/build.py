"""Geração de features derivadas."""
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona features derivadas ao Iris.

    Features novas:
        - sepal_area = sepal_length * sepal_width
        - petal_area = petal_length * petal_width
        - petal_sepal_length_ratio = petal_length / sepal_length
        - petal_sepal_width_ratio = petal_width / sepal_width
    """
    df = df.copy()
    df["sepal_area"] = df["sepal_length"] * df["sepal_width"]
    df["petal_area"] = df["petal_length"] * df["petal_width"]
    df["petal_sepal_length_ratio"] = df["petal_length"] / df["sepal_length"]
    df["petal_sepal_width_ratio"] = df["petal_width"] / df["sepal_width"]

    logger.info(f"Features derivadas adicionadas. Shape: {df.shape}")
    return df
