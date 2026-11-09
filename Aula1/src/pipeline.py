"""Pipeline end-to-end de ML.

Orquestra: ingest -> validate -> train -> evaluate -> save.

Execução:
    cd src
    python pipeline.py
"""
import logging
from pathlib import Path

import joblib

from data.ingest import load_data
from data.validate import validate_data
from train.train import train_model
from train.evaluate import evaluate_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("pipeline")


def run_pipeline():
    """Executa pipeline completo."""
    logger.info("🚀 Iniciando pipeline...")

    # 1. Ingest
    X, y = load_data()

    # 2. Validate
    validate_data(X, y)

    # 3. Train
    model, X_test, y_test = train_model(X, y)

    # 4. Evaluate
    metrics = evaluate_model(model, X_test, y_test)

    # 5. Deploy (salvar localmente)
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, "models/iris_model.pkl")
    logger.info("💾 Modelo salvo em models/iris_model.pkl")

    logger.info(f"🎉 Pipeline concluído! Accuracy: {metrics['accuracy']:.3f}")
    return metrics


if __name__ == "__main__":
    run_pipeline()
