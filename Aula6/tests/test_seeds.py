"""Testes de reprodutibilidade via seeds (Vídeo 6.2)."""
import numpy as np

from src.ml_pipeline.seeds import set_global_seeds
from src.ml_pipeline.pipeline import train


def test_set_global_seeds_retorna_seed():
    assert set_global_seeds(123) == 123


def test_mesma_seed_gera_numeros_iguais():
    set_global_seeds(42)
    a = np.random.rand(5)
    set_global_seeds(42)
    b = np.random.rand(5)
    assert np.array_equal(a, b)


def test_treino_e_reprodutivel_com_mesma_seed():
    set_global_seeds(42)
    r1 = train(model_name="rf", seed=42)
    set_global_seeds(42)
    r2 = train(model_name="rf", seed=42)
    assert r1["accuracy"] == r2["accuracy"]
    assert r1["f1_macro"] == r2["f1_macro"]
