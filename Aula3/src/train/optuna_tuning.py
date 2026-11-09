"""Hyperparameter tuning com Optuna."""
import logging

import mlflow
import optuna
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

logger = logging.getLogger(__name__)


def _objective(trial: optuna.Trial) -> float:
    """Função-objetivo: Optuna maximiza este valor."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 10, 300),
        "max_depth": trial.suggest_int("max_depth", 2, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
    }

    model = RandomForestClassifier(random_state=42, **params)
    X, y = load_iris(return_X_y=True)
    return cross_val_score(model, X, y, cv=5, scoring="accuracy").mean()


def run_optuna(n_trials: int = 30, experiment_name: str = "iris-optuna"):
    """Executa otimização Optuna e loga melhor resultado no MLflow."""
    mlflow.set_experiment(experiment_name)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(direction="maximize")
    study.optimize(_objective, n_trials=n_trials)

    logger.info(f"Optuna - Melhores: {study.best_params}")
    logger.info(f"Optuna - Score: {study.best_value:.3f}")

    with mlflow.start_run(run_name="optuna_best"):
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_cv_score", study.best_value)
        mlflow.log_param("n_trials", n_trials)

    return study


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_optuna()
