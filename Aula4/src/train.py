"""Treina e salva modelo Iris para uso no deploy."""
import logging
from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger("train")


def main():
    logger.info("Treinando modelo Iris...")
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    logger.info(f"Accuracy: {accuracy:.3f}")

    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, "models/iris_model.pkl")
    logger.info("💾 Modelo salvo em models/iris_model.pkl")


if __name__ == "__main__":
    main()
