# 🏗️ Arquitetura — Aula 06: SOLID, Reprodutibilidade e Qualidade de Código

A **Aula 06** aplica engenharia de software clássica ao pipeline de ML: **refatora**
com SOLID (reduzindo complexidade ciclomática), garante **reprodutibilidade** com
seeds + Model Registry, e impõe **qualidade de código** com PyLint e cProfile.

---

## 🎯 Visão Geral

```
   Vídeo 6.1                Vídeo 6.2                   Vídeo 6.3
+------------+          +-------------------+        +-------------------+
| SOLID      |          | Seeds + MLflow    |        | PyLint + cProfile |
| refatora   |  ----->  | Model Registry    | -----> | lint + profiling  |
| (radon)    |          | (reprodutível)    |        | (gate de CI)      |
+------------+          +-------------------+        +-------------------+
 D(23) -> A(1.6)         binario<->codigo<->deps       10/10 + sem gargalo
```

| Etapa | Ferramenta | Arquivo |
|-------|-----------|---------|
| **Refatoração SOLID** | radon (medição) | `src/ml_pipeline/*` |
| **Reprodutibilidade** | seeds + MLflow | `src/ml_pipeline/seeds.py`, `scripts/train_register.py` |
| **Model Registry** | MLflow | `scripts/train_register.py` |
| **Lint** | PyLint | `.pylintrc` |
| **Profiling** | cProfile | `scripts/profile_training.py` |

---

## 📁 Estrutura de Diretórios

```
fiap-mlet-mlops-aula06/
├── .gitignore
├── .pylintrc                    # Config do PyLint
├── README.md
├── requirements.txt
├── pytest.ini
├── docs/
│   ├── ARCHITECTURE.md
│   ├── CHEATSHEET.md
│   ├── HANDS-ON-06-01.md        # SOLID
│   ├── HANDS-ON-06-02.md        # Reprodutibilidade + Model Registry
│   └── HANDS-ON-06-03.md        # PyLint + cProfile
├── src/
│   ├── before/
│   │   └── train_monolitico.py     # contraexemplo (alta complexidade)
│   └── ml_pipeline/
│       ├── seeds.py
│       ├── data.py                 # load_data, clean_data
│       ├── model.py                # build_model + dict MODELS
│       └── pipeline.py             # train()
├── scripts/
│   ├── train_register.py
│   └── profile_training.py
├── tests/
│   ├── test_pipeline.py
│   └── test_seeds.py
└── archive/
    └── qualidade-versionamento-dados/   # material da ementa antiga (DVC/GE/drift)
```

---

## 🔄 Fluxo de Dependências (SOLID)

O sentido das setas mostra "depende de". `pipeline.train()` chama funções pequenas,
cada uma com uma responsabilidade.

```
                  +-----------------+
                  |  pipeline.train |   (pipeline.py)
                  +--------+--------+
                           | chama na ordem
       +----------+--------+---------+-----------+
       v          v                  v           v
   load_data   clean_data        build_model   (split + fit + score)
   (data.py)   (data.py)         (model.py)
                                      |
                                 dict MODELS
                                 +-- rf
                                 +-- logreg   <- adicionar = +1 linha
```

- **SRP**: cada função faz uma coisa (carregar, limpar, construir modelo, treinar).
- **OCP**: novo modelo = nova entrada no dicionário `MODELS`; `train()` não muda.

---

## 🔁 Fluxo de Reprodutibilidade (6.2)

```
set_global_seeds(42)
        |
        v
  train() ---> modelo treinado (determinístico)
        |
        v
  mlflow.start_run()
        |
        +--> log_params (model, test_size, seed)
        +--> set_tag    (git_sha)            <- código exato
        +--> log_metrics(accuracy, f1)
        +--> log_model  (pickle + requirements)  <- binario + deps
        |
        v
  register_model("iris-classifier") ---> Version N no Model Registry
```

O tripé **binário ↔ código ↔ dependências** é o que torna um modelo auditável e
reproduzível meses depois.

---

## 🧱 Decisões de Design

### 1. Manter o contraexemplo no repo
`src/before/train_monolitico.py` fica versionado de propósito: é o "antes" que dá
sentido às métricas do "depois". É ignorado pelo PyLint via `[DESIGN]`/exemplo.

### 2. Modelos num dicionário (`MODELS`)
Mais simples que herança/Factory clássico e suficiente para OCP: adicionar um modelo
é acrescentar uma linha. Cada entrada já fixa `random_state`, amarrando 6.1 e 6.2.

### 3. Funções em vez de classes
O pipeline é um conjunto de funções pequenas (`load_data`, `clean_data`, `build_model`,
`train`). Sem `Protocol`, dataclasses ou injeção de dependência — o objetivo da aula é
SRP/OCP de forma visual, não arquitetura avançada. Cada função é testável isolada.

### 4. MLflow 2.9.2 (pin)
API estável e bastante material online. Em 3.x, `artifact_path` vira `name`.

### 5. Seed em três lugares
`PYTHONHASHSEED` + `random`/`numpy` + `random_state` do estimador. Os três são
necessários para determinismo de ponta a ponta.

### 6. PyLint como gate (`--fail-under`)
Transforma qualidade em check binário de CI, não em revisão subjetiva.

---

## 🆚 Analogia com Engenharia de Software Clássica

| MLOps (aqui) | Engenharia de Software |
|--------------|------------------------|
| Refatorar código monolítico | Refatorar um God Object |
| radon / complexidade ciclomática | Métricas de manutenibilidade |
| Seeds | Testes determinísticos |
| MLflow Model Registry | Artifact repository + versionamento |
| git_sha no run | Rastreabilidade build ↔ commit |
| PyLint `--fail-under` | Quality gate no CI |
| cProfile | Profiler (APM em dev-time) |

---

## 🚀 Próximo Passo

Com o pipeline limpo, reprodutível e registrado, as próximas aulas seguem o ciclo de
MLOps sobre uma base de código que a equipe consegue manter sem medo.
