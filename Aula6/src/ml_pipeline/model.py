"""Escolha do modelo (Vídeo 6.1 - OCP).

OCP (Open/Closed): para adicionar um modelo novo, basta acrescentar UMA linha no
dicionário MODELS. O resto do código (treino, avaliação) não muda.
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Adicionar um modelo = adicionar uma linha aqui. Só isso.
MODELS = {
    "rf": lambda seed: RandomForestClassifier(n_estimators=100, random_state=seed),
    "rf_small": lambda seed: RandomForestClassifier(n_estimators=10, random_state=seed),
    "logreg": lambda seed: LogisticRegression(max_iter=200, random_state=seed),
}


def build_model(name: str, seed: int = 42):
    """Cria o modelo pelo nome, já com a seed fixada."""
    if name not in MODELS:
        raise ValueError(f"modelo desconhecido: {name}. Opções: {list(MODELS)}")
    return MODELS[name](seed)
