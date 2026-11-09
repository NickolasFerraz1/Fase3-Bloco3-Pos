# Aula 02 - Ingestão de Dados e Feature Engineering

> **Repositório**: https://github.com/josenetoo/fiap-mlet-mlops-aula02

## 🎯 Objetivo

Validar qualidade dos dados, transformar features (scaling + encoding), criar features derivadas e introduzir versionamento de dados com DVC.

> 💡 **O que é Feature Engineering?** É o trabalho de **preparar e criar boas colunas (features)**
> para o modelo aprender melhor — em vez de usar o dado cru. É como preparar bem os ingredientes
> antes de cozinhar. Tem 3 formas, cobertas nesta aula: **transformar** (vídeo 02), **selecionar**
> (vídeo 02) e **criar** features novas (vídeo 03).

## 📹 Vídeos desta Aula

| Vídeo | Tema | O que você vai fazer |
|-------|------|---------------------|
| 01 | Qualidade de Dados e Validação | Validar schema, estatísticas e detectar outliers |
| 02 | Transformação e Seleção de Features | Normalização, encoding e Pipeline sklearn |
| 03 | Feature Engineering e Versionamento | Features derivadas, persistência e DVC |

## 🏗️ Pipeline desta Aula

```
┌──────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌────────┐
│ Load │──▶│ Validate │──▶│ Outliers │──▶│ Derived  │──▶│ Persist│
└──────┘   └──────────┘   └──────────┘   │ Features │   │ + DVC  │
                                          └──────────┘   └────────┘
```

Detalhes em [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## 📁 Estrutura do Repositório

```
.
├── .gitignore
├── README.md
├── requirements.txt
├── pytest.ini
├── docs/
│   ├── ARCHITECTURE.md
│   ├── CHEATSHEET.md
│   ├── HANDS-ON-02-01.md     # Validação
│   ├── HANDS-ON-02-02.md     # Transformação
│   └── HANDS-ON-02-03.md     # Features derivadas + DVC
├── src/
│   ├── data/
│   │   ├── load.py
│   │   └── validate.py
│   ├── features/
│   │   ├── transform.py
│   │   └── build.py
│   └── pipeline.py
├── tests/
│   ├── test_validate.py
│   ├── test_transform.py
│   └── test_build.py
└── data/processed/            # Gerado pelo pipeline (gitignored, versionado por DVC)
```

## Pré-requisitos

| Requisito | Como verificar |
|-----------|----------------|
| Aula 01 concluída | Pipeline da Aula 01 rodando |
| Python 3.11 | `python3.11 --version` |
| Git | `git --version` |
| Espaço para DVC | ~50 MB livres |

## 🚀 Como Usar

1. **Fork** e clone este repositório
2. Crie venv e instale: `python3.11 -m venv venv && source venv/bin/activate && pip install -r requirements.txt`
3. Siga os hands-on em `docs/HANDS-ON-02-*.md`
4. Use código em `src/` como referência

## 📚 Documentação

| Vídeo | Hands-on |
|-------|----------|
| 01 - Qualidade de Dados e Validação | [HANDS-ON-02-01.md](docs/HANDS-ON-02-01.md) |
| 02 - Transformação e Seleção de Features | [HANDS-ON-02-02.md](docs/HANDS-ON-02-02.md) |
| 03 - Feature Engineering e Versionamento | [HANDS-ON-02-03.md](docs/HANDS-ON-02-03.md) |

**Referência rápida**: [Cheatsheet](docs/CHEATSHEET.md)

---

**FIAP - Pós Tech Machine Learning Engineering**
