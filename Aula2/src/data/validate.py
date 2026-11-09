"""Validação de qualidade de dados."""
import logging

import pandas as pd

logger = logging.getLogger(__name__)

EXPECTED_SCHEMA = {
    "sepal_length": "float64",
    "sepal_width": "float64",
    "petal_length": "float64",
    "petal_width": "float64",
    "target": "int64",
}


def validate_schema(df: pd.DataFrame) -> bool:
    """Valida colunas e tipos do DataFrame.

    Raises:
        AssertionError: Se schema inválido.
    """
    logger.info("Validando schema...")

    missing = set(EXPECTED_SCHEMA) - set(df.columns)
    assert not missing, f"Colunas faltando: {missing}"

    for col, expected_type in EXPECTED_SCHEMA.items():
        actual_type = str(df[col].dtype)
        assert actual_type == expected_type, (
            f"Tipo errado em '{col}': esperado {expected_type}, obtido {actual_type}"
        )

    logger.info(f"✅ Schema válido: {len(df)} linhas x {len(df.columns)} colunas")
    return True


def detect_outliers_iqr(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Detecta outliers usando IQR (Interquartile Range)."""
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = df[(df[column] < lower) | (df[column] > upper)]
    logger.info(f"Outliers em '{column}': {len(outliers)} / {len(df)}")
    return outliers
