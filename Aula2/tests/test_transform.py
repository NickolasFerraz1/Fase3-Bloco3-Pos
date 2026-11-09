"""Testes para features/transform.py."""
import pandas as pd

from features.transform import build_full_preprocessor, build_numeric_pipeline


def test_numeric_pipeline_normaliza():
    pipe = build_numeric_pipeline()
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]})
    X_scaled = pipe.fit_transform(X)
    # Após StandardScaler, médias ≈ 0
    assert abs(X_scaled.mean()) < 1e-9


def test_pipeline_processa_dataframe_misto():
    df = pd.DataFrame({
        "idade": [25, 45, 35],
        "salario": [3000, 8000, 5000],
        "cidade": ["SP", "RJ", "MG"],
    })

    preprocessor = build_full_preprocessor(
        numeric_cols=["idade", "salario"],
        categorical_cols=["cidade"],
    )

    X = preprocessor.fit_transform(df)
    # 2 numéricas + 3 cidades
    assert X.shape == (3, 5)
