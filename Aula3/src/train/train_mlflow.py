"""Treinamento com rastreamento MLflow."""
import logging

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier

from train.metrics import evaluate
from train.split import get_train_test_split

logger = logging.getLogger(__name__)


def train_run(
    n_estimators: int = 100,
    max_depth=None,
    experiment_name: str = "iris-classification",
):
    """Treina modelo e loga tudo no MLflow."""
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        X_train, X_test, y_train, y_test = get_train_test_split()

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
        )
        model.fit(X_train, y_train)

        metrics, cm = evaluate(model, X_test, y_test)

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", str(max_depth))
        mlflow.log_param("model_type", "RandomForest")

        for name, value in metrics.items():
            mlflow.log_metric(name, value)

        mlflow.sklearn.log_model(model, "model")

        logger.info(f"✅ Run logado. Accuracy: {metrics['accuracy']:.3f}")
        return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_run()
