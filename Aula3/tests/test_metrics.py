"""Testes para train/metrics.py."""
from sklearn.ensemble import RandomForestClassifier

from train.metrics import evaluate
from train.split import get_train_test_split


def _trained_model():
    X_tr, X_te, y_tr, y_te = get_train_test_split()
    model = RandomForestClassifier(n_estimators=50, random_state=42).fit(X_tr, y_tr)
    return model, X_te, y_te


def test_evaluate_retorna_metrics_e_matriz():
    model, X_te, y_te = _trained_model()
    metrics, cm = evaluate(model, X_te, y_te)
    assert isinstance(metrics, dict)
    assert cm.shape == (3, 3)  # 3 classes Iris


def test_metricas_principais_existem():
    model, X_te, y_te = _trained_model()
    metrics, _ = evaluate(model, X_te, y_te)
    for key in ("accuracy", "precision", "recall", "f1"):
        assert key in metrics


def test_accuracy_acima_de_threshold():
    model, X_te, y_te = _trained_model()
    metrics, _ = evaluate(model, X_te, y_te)
    assert metrics["accuracy"] >= 0.9
