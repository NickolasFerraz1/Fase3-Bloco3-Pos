# Aula 03 - Treinamento de Modelos e Validação

> **Repositório**: https://github.com/josenetoo/fiap--mlet-mlops-aula03

## 🎯 Objetivo

Treinar modelos com **split estratificado**, avaliar com **métricas robustas** (precision/recall/F1), rastrear experimentos com **MLflow** e fazer **tuning automático** com GridSearch, RandomSearch e Optuna.

## 📹 Vídeos desta Aula

| Vídeo | Tema | O que você vai fazer |
|-------|------|---------------------|
| 01 | Generalização, Overfitting e Métricas | Split estratificado, métricas além de accuracy, matriz de confusão |
| 02 | Treino e Rastreabilidade com MLflow | Logar runs, comparar na UI, Model Registry (Staging → Production) |
| 03 | Validação Cruzada e Tuning | K-Fold, GridSearch, RandomSearch, Optuna |

## 🏗️ Pipeline desta Aula

```
┌───────┐   ┌────────┐   ┌─────────┐   ┌────────────┐   ┌────────────────┐
│ Split │──▶│ Train  │──▶│ Metrics │──▶│ MLflow Log │──▶│ Tuning + Registry │
└───────┘   └────────┘   └─────────┘   └────────────┘   └────────────────┘
```

Detalhes em [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## � Estrutura do Repositório

```
.
├── .gitignore
├── README.md
├── requirements.txt
├── pytest.ini
├── docs/
│   ├── ARCHITECTURE.md
│   ├── CHEATSHEET.md
│   ├── HANDS-ON-03-01.md      # Métricas e Overfitting
│   ├── HANDS-ON-03-02.md      # MLflow
│   └── HANDS-ON-03-03.md      # Tuning
├── src/
│   └── train/
│       ├── split.py
│       ├── metrics.py
│       ├── demo_overfit.py
│       ├── train_mlflow.py
│       ├── cross_val.py
│       ├── tuning.py
│       └── optuna_tuning.py
├── tests/
│   ├── test_split.py
│   └── test_metrics.py
└── mlruns/                    # MLflow tracking (gitignored)
```

## Pré-requisitos

| Requisito | Como verificar |
|-----------|----------------|
| Aula 02 concluída | Pipeline da Aula 02 rodando |
| Python 3.11 | `python3.11 --version` |
| Porta 5001 livre | `lsof -i :5001` (Mac/Linux) |

## 🚀 Como Usar

1. **Fork** e clone este repositório
2. Crie venv e instale: `python3.11 -m venv venv && source venv/bin/activate && pip install -r requirements.txt`
3. Em **um terminal**: `mlflow ui --port 5001`
4. Em **outro terminal**: siga os hands-on
5. Use código em `src/train/` como referência

## 📚 Documentação

| Vídeo | Hands-on |
|-------|----------|
| 01 - Generalização, Overfitting e Métricas | [HANDS-ON-03-01.md](docs/HANDS-ON-03-01.md) |
| 02 - Treino e Rastreabilidade com MLflow | [HANDS-ON-03-02.md](docs/HANDS-ON-03-02.md) |
| 03 - Validação Cruzada e Tuning | [HANDS-ON-03-03.md](docs/HANDS-ON-03-03.md) |

**Referência rápida**: [Cheatsheet](docs/CHEATSHEET.md)

---

**FIAP - Pós Tech Machine Learning Engineering**
