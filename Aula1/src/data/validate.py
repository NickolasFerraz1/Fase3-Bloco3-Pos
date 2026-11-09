"""Validação de qualidade dos dados."""
import logging
import numpy as np

logger = logging.getLogger(__name__)


def validate_data(X, y):
    """Valida que os dados estão prontos para treino.

    Args:
        X: Matriz de features.
        y: Vetor de labels.

    Returns:
        bool: True se válido.

    Raises:
        AssertionError: Se alguma validação falhar.
    """
    logger.info("Validando dados...")

    assert len(X) > 0, "Dataset vazio!"
    assert len(X) == len(y), "X e y com tamanhos diferentes!"
    assert not np.isnan(X).any(), "X contém NaN!"
    assert len(np.unique(y)) >= 2, "Menos de 2 classes!"

    logger.info(f"✅ Dados válidos: {len(X)} amostras, {len(np.unique(y))} classes")
    return True
