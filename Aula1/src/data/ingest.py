"""Ingestão de dados - carrega dataset Iris."""
import logging
from sklearn.datasets import load_iris

logger = logging.getLogger(__name__)


def load_data():
    """Carrega dataset Iris e retorna (X, y).

    Returns:
        tuple: (X, y) onde X é matriz de features e y é vetor de labels.
    """
    logger.info("Carregando dataset Iris...")
    X, y = load_iris(return_X_y=True)
    logger.info(f"Dataset carregado: {X.shape[0]} amostras, {X.shape[1]} features")
    return X, y
