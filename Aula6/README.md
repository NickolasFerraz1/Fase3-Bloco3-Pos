# Aula 06 - SOLID, Reprodutibilidade e Qualidade de Código

> **Repositório**: https://github.com/josenetoo/fiap-mlet-mlops-aula06

## 🎯 Objetivo

Aplicar engenharia de software clássica em projetos de ML: refatorar código com
**princípios SOLID** e reduzir complexidade ciclomática, garantir **reprodutibilidade**
com seeds e **Model Registry** (MLflow), e fechar o ciclo de **qualidade de código**
com **PyLint** e **cProfile**.

## 📹 Vídeos desta Aula

| Vídeo | Tema | O que você vai fazer |
|-------|------|---------------------|
| 01 | Princípios SOLID em Projetos de ML | Medir complexidade ciclomática e refatorar uma função monolítica |
| 02 | Reprodutibilidade e Model Registry | Fixar seeds, logar no MLflow e registrar o modelo |
| 03 | Qualidade com PyLint e cProfile | Padronizar com PyLint, achar gargalos com cProfile |

## 🏗️ Pipeline desta Aula

```
  Vídeo 6.1                Vídeo 6.2                  Vídeo 6.3
+-----------+          +------------------+        +------------------+
|  SOLID    |          | Seeds + MLflow   |        | PyLint + cProfile|
| refatora  | -------> | Model Registry   | -----> | lint + profiling |
| (radon)   |          | (reprodutível)   |        | (qualidade)      |
+-----------+          +------------------+        +------------------+
 complexidade            rastreabilidade             padronização
   baixa                 código<->modelo<->deps        + performance
```

Detalhes em [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## 📁 Estrutura do Repositório

```
.
├── .gitignore
├── .pylintrc                     # Config do PyLint (Vídeo 6.3)
├── README.md
├── requirements.txt
├── pytest.ini
├── docs/
│   ├── ARCHITECTURE.md
│   ├── CHEATSHEET.md
│   ├── HANDS-ON-06-01.md         # SOLID + complexidade ciclomática
│   ├── HANDS-ON-06-02.md         # Seeds + MLflow Model Registry
│   └── HANDS-ON-06-03.md         # PyLint + cProfile
├── src/
│   ├── before/
│   │   └── train_monolitico.py   # função monolítica (ponto de partida do 6.1)
│   └── ml_pipeline/              # versão refatorada (SRP + OCP)
│       ├── seeds.py              # fixação de seeds (6.2)
│       ├── data.py               # load_data, clean_data (SRP)
│       ├── model.py              # build_model + dict MODELS (OCP)
│       └── pipeline.py           # train() orquestra na ordem
├── scripts/
│   ├── train_register.py         # treina + registra no MLflow (6.2)
│   └── profile_training.py       # cProfile (6.3)
├── tests/
│   ├── test_pipeline.py          # testa load/clean/build/train
│   └── test_seeds.py             # testa reprodutibilidade
└── archive/                      # material da ementa antiga (DVC/GE/drift)
```

## Pré-requisitos

| Requisito | Como verificar |
|-----------|----------------|
| Python 3.11 | `python3.11 --version` |
| Git | `git --version` |
| Aulas anteriores | Familiaridade com sklearn/MLflow |

## 🚀 Como Usar

1. **Fork** e clone este repositório
2. Crie e ative o ambiente virtual:

**Linux/Mac:**
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

3. Siga os hands-on em `docs/HANDS-ON-06-*.md`

Comandos rápidos:

**Linux/Mac:**
```bash
# 6.1 - medir complexidade
radon cc src/before/train_monolitico.py -s
radon cc src/ml_pipeline/ -s -a

# 6.2 - treinar e registrar no MLflow
PYTHONPATH=. python scripts/train_register.py --model rf
mlflow ui --port 5001   # abre http://127.0.0.1:5001 (5000 é ocupada no macOS)

# 6.3 - qualidade de código
pylint src/ml_pipeline/
PYTHONPATH=. python scripts/profile_training.py

# Testes
pytest
```

**Windows (PowerShell):**
```powershell
# 6.1 - medir complexidade
radon cc src/before/train_monolitico.py -s
radon cc src/ml_pipeline/ -s -a

# 6.2 - treinar e registrar no MLflow
$env:PYTHONPATH = "."
python scripts/train_register.py --model rf
mlflow ui --port 5001   # abre http://127.0.0.1:5001 (5000 é ocupada no macOS)

# 6.3 - qualidade de código
pylint src/ml_pipeline/
python scripts/profile_training.py

# Testes
pytest
```

## 📚 Documentação

| Vídeo | Hands-on |
|-------|----------|
| 01 - Princípios SOLID em Projetos de ML | [HANDS-ON-06-01.md](docs/HANDS-ON-06-01.md) |
| 02 - Reprodutibilidade e Model Registry | [HANDS-ON-06-02.md](docs/HANDS-ON-06-02.md) |
| 03 - Qualidade com PyLint e cProfile | [HANDS-ON-06-03.md](docs/HANDS-ON-06-03.md) |

**Referência rápida**: [Cheatsheet](docs/CHEATSHEET.md)

---

**FIAP - Pós Tech Machine Learning Engineering**
