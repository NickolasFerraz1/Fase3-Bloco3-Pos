"""Testes para train/train.py e train/evaluate.py."""
import pytest
from data.ingest import load_data
from train.train import train_model
from train.evaluate import evaluate_model


@pytest.fixture
def dados():
    """Fixture compartilhada."""
    return load_data()


class TestTrain:
    def test_modelo_treina_sem_erro(self, dados):
        X, y = dados
        model, X_test, y_test = train_model(X, y)
        assert model is not None

    def test_modelo_e_reproduzivel(self, dados):
        """Mesmo random_state → mesmo modelo."""
        X, y = dados
        m1, _, _ = train_model(X, y, random_state=42)
        m2, _, _ = train_model(X, y, random_state=42)
        # Mesma predição para mesmo input
        assert (m1.predict(X) == m2.predict(X)).all()


class TestEvaluate:
    def test_accuracy_acima_de_90(self, dados):
        X, y = dados
        model, X_test, y_test = train_model(X, y)
        metrics = evaluate_model(model, X_test, y_test)
        assert metrics["accuracy"] >= 0.90

    def test_retorna_dict_com_accuracy(self, dados):
        X, y = dados
        model, X_test, y_test = train_model(X, y)
        metrics = evaluate_model(model, X_test, y_test)
        assert "accuracy" in metrics
