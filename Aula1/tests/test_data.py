"""Testes para data/ingest.py e data/validate.py."""
import pytest
import numpy as np
from data.ingest import load_data
from data.validate import validate_data


class TestIngest:
    def test_load_data_retorna_X_e_y(self):
        X, y = load_data()
        assert X is not None
        assert y is not None

    def test_load_data_tamanho_correto(self):
        X, y = load_data()
        assert X.shape == (150, 4)
        assert y.shape == (150,)

    def test_load_data_3_classes(self):
        _, y = load_data()
        assert len(np.unique(y)) == 3


class TestValidate:
    def test_dados_validos_passam(self):
        X, y = load_data()
        assert validate_data(X, y) is True

    def test_dataset_vazio_falha(self):
        with pytest.raises(AssertionError, match="Dataset vazio"):
            validate_data(np.array([]).reshape(0, 4), np.array([]))

    def test_x_y_tamanhos_diferentes_falha(self):
        with pytest.raises(AssertionError, match="tamanhos diferentes"):
            validate_data(np.ones((10, 4)), np.ones(5))
