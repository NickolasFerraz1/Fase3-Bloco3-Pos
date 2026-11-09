"""Testes para train/split.py."""
import numpy as np

from train.split import get_train_test_split


def test_split_retorna_4_arrays():
    X_tr, X_te, y_tr, y_te = get_train_test_split()
    assert X_tr is not None
    assert X_te is not None
    assert y_tr is not None
    assert y_te is not None


def test_split_proporcao_correta():
    X_tr, X_te, _, _ = get_train_test_split(test_size=0.2)
    assert len(X_tr) == 120
    assert len(X_te) == 30


def test_split_estratificado():
    _, _, y_tr, y_te = get_train_test_split()
    # Mesma quantidade de cada classe em ambos
    assert np.bincount(y_tr).tolist() == [40, 40, 40]
    assert np.bincount(y_te).tolist() == [10, 10, 10]


def test_split_reproduzivel():
    X1, _, _, _ = get_train_test_split(random_state=42)
    X2, _, _, _ = get_train_test_split(random_state=42)
    assert (X1 == X2).all()
