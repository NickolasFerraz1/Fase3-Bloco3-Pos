"""PONTO DE PARTIDA (Vídeo 6.1) - código deliberadamente ruim.

Uma única função **monolítica** que faz TUDO: carrega, valida, limpa, divide, escolhe
modelo, treina, avalia, imprime relatório e salva. Alta complexidade ciclomática,
zero coesão, impossível de testar em partes.

NÃO use isto como referência. Ele existe apenas para ser medido (radon/pylint) e
depois refatorado para `src/ml_pipeline/` aplicando SOLID.
"""
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


def run_everything(model_name="rf", test_size=0.2, save_path="model.pkl", verbose=True):
    """Faz absolutamente tudo numa função só (anti-padrão proposital)."""
    # 1) Carregar dados
    if os.path.exists("data/iris.csv"):
        df = pd.read_csv("data/iris.csv")
    else:
        iris = load_iris(as_frame=True)
        df = iris.frame
        df.columns = [
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
            "target",
        ]

    # 2) Validar (manualmente, espalhado)
    if df is None or len(df) == 0:
        raise ValueError("dataset vazio")
    if "target" not in df.columns:
        raise ValueError("sem coluna target")
    for c in ["sepal_length", "sepal_width", "petal_length", "petal_width"]:
        if c not in df.columns:
            raise ValueError("coluna faltando: " + c)

    # 3) Limpar (mais ramos)
    if df.isnull().values.any():
        for c in df.columns:
            if df[c].dtype == "float64" or df[c].dtype == "int64":
                df[c] = df[c].fillna(df[c].mean())
            else:
                df[c] = df[c].fillna("missing")
    if df.duplicated().any():
        df = df.drop_duplicates()

    # 4) Remover outliers grosseiros, coluna a coluna
    for c in ["sepal_length", "sepal_width", "petal_length", "petal_width"]:
        lo = df[c].quantile(0.01)
        hi = df[c].quantile(0.99)
        keep = []
        for v in df[c].values:
            if v < lo:
                keep.append(False)
            elif v > hi:
                keep.append(False)
            else:
                keep.append(True)
        df = df[np.array(keep)]

    # 5) Split
    x = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    y = df["target"]
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=test_size)

    # 6) Escolher e treinar modelo (mais ramos)
    if model_name == "rf":
        model = RandomForestClassifier(n_estimators=100)
    elif model_name == "rf_small":
        model = RandomForestClassifier(n_estimators=10)
    elif model_name == "logreg":
        model = LogisticRegression(max_iter=200)
    else:
        raise ValueError("modelo desconhecido: " + str(model_name))
    model.fit(x_tr, y_tr)

    # 7) Avaliar
    pred = model.predict(x_te)
    acc = accuracy_score(y_te, pred)
    f1 = f1_score(y_te, pred, average="macro")

    # 8) Relatório
    if verbose:
        print("=== RELATORIO ===")
        print("modelo:", model_name)
        print("linhas:", len(df))
        print("accuracy:", round(acc, 4))
        print("f1_macro:", round(f1, 4))
        if acc > 0.9:
            print("status: OK")
        elif acc > 0.7:
            print("status: ATENCAO")
        else:
            print("status: RUIM")

    # 9) Salvar
    if save_path:
        with open(save_path, "wb") as f:
            pickle.dump(model, f)

    return {"accuracy": acc, "f1_macro": f1, "n_rows": len(df)}


if __name__ == "__main__":
    run_everything()
