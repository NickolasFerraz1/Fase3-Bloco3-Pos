"""Pipeline de feature engineering end-to-end.

Fluxo: load -> validate -> detect outliers -> build derived features -> persist.

Execução (a partir da raiz do projeto):
    python src/pipeline.py
"""
import logging
from pathlib import Path

from data.load import load_iris_df
from data.validate import detect_outliers_iqr, validate_schema
from features.build import add_derived_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("pipeline")


def run():
    """Executa pipeline de features."""
    logger.info("🚀 Pipeline de features iniciado")

    # 1. Carregar
    df = load_iris_df()

    # 2. Validar schema
    validate_schema(df)

    # 3. Detectar outliers (apenas log; no Iris não removemos)
    for col in ["sepal_length", "sepal_width", "petal_length", "petal_width"]:
        detect_outliers_iqr(df, col)

    # 4. Features derivadas
    df = add_derived_features(df)

    # 5. Persistir
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "iris_features.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"💾 Dataset salvo em {out_path}")

    logger.info(f"🎉 Pipeline concluído. Shape final: {df.shape}")
    return df


if __name__ == "__main__":
    run()
