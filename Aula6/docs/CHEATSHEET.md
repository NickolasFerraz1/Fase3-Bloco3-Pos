# 📋 CHEATSHEET - Aula 06: SOLID, Reprodutibilidade e Qualidade de Código

## Princípios SOLID (foco da aula: SRP e OCP)

| Princípio | Em uma frase | Onde no projeto |
|-----------|--------------|-----------------|
| **S**RP | Uma tarefa por função | `data.py` (`load_data`, `clean_data`) |
| **O**CP | Adicionar = +1 linha, sem reescrever | dict `MODELS` em `model.py` |
| **L**SP | Subtipos substituem o tipo base | (conceito; não focado aqui) |
| **I**SP | Interfaces pequenas e específicas | (conceito; não focado aqui) |
| **D**IP | Dependa de abstrações | (conceito; não focado aqui) |

> Na aula focamos **SRP** e **OCP** — os mais visuais. Os outros três ficam como
> referência conceitual.

## Complexidade Ciclomática (radon)

```bash
# Complexidade por bloco, com score A-F
radon cc src/ -s

# Com média ao final
radon cc src/ml_pipeline/ -s -a

# Índice de manutenibilidade
radon mi src/ml_pipeline/
```

| Score | Faixa | Leitura |
|-------|-------|---------|
| A | 1-5 | simples |
| B-C | 6-20 | atenção |
| D-F | 21+ | refatorar |

## Reprodutibilidade (seeds)

```python
import os, random
import numpy as np

def set_global_seeds(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    return seed

# E no estimador (separado!):
RandomForestClassifier(random_state=seed)
train_test_split(x, y, random_state=seed)
```

## MLflow Tracking + Model Registry

```python
import mlflow, mlflow.sklearn

mlflow.set_experiment("aula06-model-registry")

with mlflow.start_run() as run:
    mlflow.log_params({"model": "rf", "seed": 42})
    mlflow.set_tag("git_sha", "<commit>")
    mlflow.log_metrics({"accuracy": 0.93, "f1_macro": 0.93})
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",              # MLflow 3.x: use name=
        registered_model_name="iris-classifier",
    )
```

```bash
# UI (terminal separado) - porta 5001 porque a 5000 é ocupada no macOS
mlflow ui --port 5001                       # http://127.0.0.1:5001

# Com backend de banco (Registry completo)
mlflow ui --port 5001 --backend-store-uri sqlite:///mlflow.db
```

```python
# Promover versão entre stages
from mlflow import MlflowClient
MlflowClient().transition_model_version_stage(
    name="iris-classifier", version=1, stage="Production"
)
```

## PyLint

```bash
pylint src/ml_pipeline/                # nota 0-10
pylint src/ml_pipeline/ --fail-under=9.0   # gate de CI (exit != 0 reprova)
pylint --generate-rcfile > .pylintrc   # gerar config base
```

Mensagens comuns: `C` convenção, `R` refactor, `W` warning, `E` erro.

```ini
# .pylintrc (trechos úteis)
[MESSAGES CONTROL]
disable=missing-module-docstring,too-few-public-methods
[DESIGN]
max-branches=12
[FORMAT]
max-line-length=100
```

## cProfile

```bash
# Perfil rápido pela linha de comando, ordenado por tempo cumulativo
python -m cProfile -s cumulative scripts/profile_training.py

# No código
python scripts/profile_training.py --top 15
python scripts/profile_training.py --dump train.prof
snakeviz train.prof                    # visual (terminal separado)
```

| Coluna | Significado |
|--------|-------------|
| `ncalls` | nº de chamadas |
| `tottime` | tempo só na função (sem filhas) — **ache o gargalo aqui** |
| `cumtime` | tempo incluindo chamadas filhas |

## Testes

```bash
pytest                       # tudo
pytest tests/test_seeds.py -v
pytest --cov=src             # com cobertura (pytest-cov)
```

## Fluxo de Qualidade no CI

```
radon cc (informativo)
        |
   pylint --fail-under=9.0  --(reprova)--> bloqueia
        |
   pytest  --------------- --(reprova)--> bloqueia
        |
      merge
```
