"""Transformações de features (scaling, encoding)."""
import logging

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


def build_numeric_pipeline() -> Pipeline:
    """Pipeline para features numéricas: imputação -> normalização."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])


def build_categorical_pipeline() -> Pipeline:
    """Pipeline para features categóricas: one-hot encoding."""
    return Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])


def build_full_preprocessor(numeric_cols, categorical_cols) -> ColumnTransformer:
    """Pré-processador completo: aplica transformação diferente por tipo de coluna."""
    return ColumnTransformer([
        ("num", build_numeric_pipeline(), numeric_cols),
        ("cat", build_categorical_pipeline(), categorical_cols),
    ])
