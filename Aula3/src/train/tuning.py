"""Hyperparameter tuning com Grid e Random Search."""
import logging

import mlflow
from scipy.stats import randint
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

logger = logging.getLogger(__name__)


def grid_search(experiment_name: str = "iris-tuning"):
    """GridSearchCV: testa todas as combinações."""
    mlflow.set_experiment(experiment_name)
    X, y = load_iris(return_X_y=True)

    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 10, None],
        "min_samples_split": [2, 5],
    }

    with mlflow.start_run(run_name="grid_search"):
        search = GridSearchCV(
            estimator=RandomForestClassifier(random_state=42),
            param_grid=param_grid,
            cv=5,
            scoring="accuracy",
            n_jobs=-1,
        )
        search.fit(X, y)

        logger.info(f"Grid - Melhores: {search.best_params_}")
        logger.info(f"Grid - Score: {search.best_score_:.3f}")

        mlflow.log_params({f"best_{k}": v for k, v in search.best_params_.items()})
        mlflow.log_metric("best_cv_score", search.best_score_)
        mlflow.sklearn.log_model(search.best_estimator_, "model")

        return search.best_estimator_, search.best_params_


def random_search(n_iter: int = 20, experiment_name: str = "iris-tuning"):
    """RandomizedSearchCV: amostragem aleatória."""
    mlflow.set_experiment(experiment_name)
    X, y = load_iris(return_X_y=True)

    param_dist = {
        "n_estimators": randint(10, 300),
        "max_depth": randint(2, 20),
        "min_samples_split": randint(2, 10),
    }

    with mlflow.start_run(run_name="random_search"):
        search = RandomizedSearchCV(
            estimator=RandomForestClassifier(random_state=42),
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=5,
            scoring="accuracy",
            random_state=42,
            n_jobs=-1,
        )
        search.fit(X, y)

        logger.info(f"Random - Melhores: {search.best_params_}")
        logger.info(f"Random - Score: {search.best_score_:.3f}")

        mlflow.log_params({f"best_{k}": v for k, v in search.best_params_.items()})
        mlflow.log_metric("best_cv_score", search.best_score_)

        return search.best_estimator_, search.best_params_


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    grid_search()
