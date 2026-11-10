#import os  # <- linha temporária só pra demo
"""Orquestração do treino (Vídeo 6.1).

Junta as peças na ordem: carrega -> limpa -> divide -> treina -> avalia.
Cada passo é uma função pequena de `data.py`/`model.py`, fácil de ler e testar.
"""
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from src.ml_pipeline.data import FEATURES, TARGET, clean_data, load_data
from src.ml_pipeline.model import build_model


def train(model_name: str = "rf", test_size: float = 0.2, seed: int = 42) -> dict:
    """Treina um modelo e devolve o modelo + métricas.

    Returns:
        Dict com chaves: model, accuracy, f1_macro, n_rows.
    """
    df = clean_data(load_data())
    x = df[FEATURES]
    y = df[TARGET]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=seed, stratify=y
    )

    model = build_model(model_name, seed=seed)
    model.fit(x_train, y_train)

    pred = model.predict(x_test)
    return {
        "model": model,
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1_macro": float(f1_score(y_test, pred, average="macro")),
        "n_rows": len(df),
    }
