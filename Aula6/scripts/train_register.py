"""Treina, loga no MLflow e registra no Model Registry (Vídeo 6.2).

Amarra três coisas que precisam andar juntas para reprodutibilidade:
- o binário do modelo (artefato logado),
- o código exato que o gerou (git commit sha, como tag),
- as dependências/seed (params + ambiente capturado pelo mlflow.sklearn).
"""
import argparse
import subprocess

import mlflow
import mlflow.sklearn

from src.ml_pipeline.seeds import set_global_seeds
from src.ml_pipeline.pipeline import train

EXPERIMENT = "aula06-model-registry"
REGISTERED_MODEL = "iris-classifier"


def _git_sha() -> str:
    """Retorna o commit atual (ou 'unknown' fora de um repo git)."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return sha.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def train_and_register(model_name: str, test_size: float, seed: int, register: bool = True) -> str:
    """Executa o treino dentro de um run do MLflow e registra o modelo.

    Returns:
        O run_id do MLflow.
    """
    seed = set_global_seeds(seed)
    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run() as run:
        # Reprodutibilidade: params + código que gerou o modelo
        mlflow.log_params({"model_name": model_name, "test_size": test_size, "seed": seed})
        mlflow.set_tag("git_sha", _git_sha())

        result = train(model_name=model_name, test_size=test_size, seed=seed)
        mlflow.log_metrics(
            {"accuracy": result["accuracy"], "f1_macro": result["f1_macro"]}
        )

        # log_model captura o ambiente (requirements/conda) junto ao binário.
        # registered_model_name cria/atualiza uma versão no Model Registry.
        mlflow.sklearn.log_model(
            sk_model=result["model"],
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL if register else None,
        )

        print(f"✅ run_id={run.info.run_id}")
        print(f"   accuracy={result['accuracy']:.4f} f1_macro={result['f1_macro']:.4f}")
        if register:
            print(f"   modelo registrado como '{REGISTERED_MODEL}'")
        return run.info.run_id


def _parse_args():
    parser = argparse.ArgumentParser(description="Treina e registra no MLflow")
    parser.add_argument("--model", default="rf", help="rf | rf_small | logreg")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train_and_register(args.model, args.test_size, args.seed)
