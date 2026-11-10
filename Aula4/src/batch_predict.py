"""Predição em batch: lê CSV de entrada, prediz, salva CSV de saída."""
import argparse
import logging
from pathlib import Path

import joblib
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger("batch")

CLASSES = {0: "setosa", 1: "versicolor", 2: "virginica"}


def predict_batch(input_path: str, output_path: str, model_path: str) -> pd.DataFrame:
    """Lê CSV, prediz e salva resultados."""
    logger.info(f"Carregando modelo de {model_path}")
    model = joblib.load(model_path)

    logger.info(f"Lendo entrada: {input_path}")
    df = pd.read_csv(input_path)
    logger.info(f"   {len(df)} linhas para processar")

    predictions = model.predict(df.values)
    probas = model.predict_proba(df.values).max(axis=1)

    df["prediction"] = predictions
    df["class_name"] = [CLASSES[int(p)] for p in predictions]
    df["confidence"] = probas.round(3)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"💾 Resultados salvos em {output_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predição em batch")
    parser.add_argument("--input", default="data/input.csv")
    parser.add_argument("--output", default="data/predictions.csv")
    parser.add_argument("--model", default="models/iris_model.pkl")
    args = parser.parse_args()

    df = predict_batch(args.input, args.output, args.model)
    print(df.to_string(index=False))
