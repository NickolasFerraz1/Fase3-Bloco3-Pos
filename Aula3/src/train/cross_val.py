"""Validação cruzada."""
import logging

import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

logger = logging.getLogger(__name__)


def cross_validate(model=None, k: int = 5) -> np.ndarray:
    """Roda K-Fold CV e retorna array de scores."""
    if model is None:
        model = RandomForestClassifier(n_estimators=100, random_state=42)

    X, y = load_iris(return_X_y=True)
    scores = cross_val_score(model, X, y, cv=k, scoring="accuracy")

    logger.info(f"K-Fold (k={k}) scores: {scores.round(3)}")
    logger.info(f"Média: {scores.mean():.3f} ± {scores.std():.3f}")
    return scores


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cross_validate()
