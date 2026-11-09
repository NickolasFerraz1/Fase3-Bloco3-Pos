"""Cálculo de métricas de classificação."""
import logging

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


def evaluate(model, X_test, y_test):
    """Avalia modelo e retorna (dict de métricas, matriz de confusão)."""
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }

    logger.info(f"Métricas: { {k: round(v, 3) for k, v in metrics.items()} }")
    logger.info("\n" + classification_report(y_test, y_pred, zero_division=0))

    return metrics, confusion_matrix(y_test, y_pred)
