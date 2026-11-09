"""Divisão treino/teste estratificada."""
import logging

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def get_train_test_split(test_size: float = 0.2, random_state: int = 42):
    """Retorna (X_train, X_test, y_train, y_test) estratificado.

    Args:
        test_size: Proporção do conjunto de teste.
        random_state: Seed para reprodutibilidade.

    Returns:
        Tupla com 4 arrays numpy.
    """
    X, y = load_iris(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    logger.info(f"Split: treino={len(X_train)}, teste={len(X_test)}")
    return X_train, X_test, y_train, y_test
