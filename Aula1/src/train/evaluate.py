"""Avaliação do modelo."""
import logging
from sklearn.metrics import accuracy_score, classification_report

logger = logging.getLogger(__name__)


def evaluate_model(model, X_test, y_test):
    """Avalia modelo e retorna métricas.

    Args:
        model: Modelo treinado.
        X_test: Features de teste.
        y_test: Labels de teste.

    Returns:
        dict: Dicionário com métricas (accuracy).
    """
    logger.info("Avaliando modelo...")

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    logger.info(f"✅ Accuracy: {accuracy:.3f}")
    logger.info("\n" + classification_report(y_test, y_pred))

    return {"accuracy": accuracy}
