# Aula 01 - Introdução ao Pipeline de ML

> **Repositório**: https://github.com/josenetoo/fiap-mlet-mlops-aula01

## 🎯 Objetivo

Montar a base de um projeto MLOps: ambiente isolado, pipeline modular de ML (ingest → validate → train → evaluate → deploy) e testes automatizados com `pytest`.

## 📹 Vídeos desta Aula

| Vídeo | Tema | O que você vai fazer |
|-------|------|---------------------|
| 01 | Setup do Ambiente MLOps | Criar venv, instalar dependências e treinar primeiro modelo |
| 02 | Arquitetura de um Pipeline de ML | Refatorar para pipeline modular com logging |
| 03 | Testes e Reprodutibilidade | Adicionar pytest, cobertura e fixar random seeds |

## 🏗️ Arquitetura do Pipeline

```
┌─────────┐   ┌──────────┐   ┌────────┐   ┌──────────┐   ┌────────┐
│ Ingest  │──▶│ Validate │──▶│ Train  │──▶│ Evaluate │──▶│ Deploy │
└─────────┘   └──────────┘   └────────┘   └──────────┘   └────────┘
```

Detalhes em [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## 📁 Estrutura do Repositório

```
.
├── .gitignore
├── README.md
├── requirements.txt          # Dependências (pip)
├── pytest.ini                # Configuração de testes
├── config.yaml               # Parâmetros do pipeline
├── docs/
│   ├── ARCHITECTURE.md       # Diagrama e fluxo
│   ├── CHEATSHEET.md         # Comandos rápidos
│   ├── HANDS-ON-01-01.md     # Vídeo 1.1 - Setup
│   ├── HANDS-ON-01-02.md     # Vídeo 1.2 - Pipeline modular
│   └── HANDS-ON-01-03.md     # Vídeo 1.3 - Testes
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── ingest.py
│   │   └── validate.py
│   ├── train/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   └── evaluate.py
│   └── pipeline.py
├── tests/
│   ├── __init__.py
│   ├── test_data.py          # Testes de ingestão/validação
│   ├── test_train.py         # Testes de treino/avaliação
│   └── test_pipeline.py      # Teste end-to-end
└── models/                   # Modelos treinados (gitignored)
```

## Pré-requisitos

| Requisito | Como verificar |
|-----------|----------------|
| Python 3.11 ou 3.12 | `python3 --version` |
| Git | `git --version` |
| Editor de código | VS Code, PyCharm ou similar |

> ⚠️ Use Python **3.11 ou 3.12**. As versões fixadas no `requirements.txt` não têm pacotes pré-compilados para o Python 3.13+, o que faz o pip tentar compilar do zero e falhar.

## 🚀 Como Usar

1. **Fork** este repositório
2. Clone o seu fork localmente
3. Siga os arquivos hands-on em `docs/HANDS-ON-01-*.md`
4. Use o código em `src/` como referência

## 📚 Documentação

| Vídeo | Hands-on |
|-------|----------|
| 01 - Setup do Ambiente MLOps | [HANDS-ON-01-01.md](docs/HANDS-ON-01-01.md) |
| 02 - Arquitetura de um Pipeline de ML | [HANDS-ON-01-02.md](docs/HANDS-ON-01-02.md) |
| 03 - Testes e Reprodutibilidade | [HANDS-ON-01-03.md](docs/HANDS-ON-01-03.md) |

**Referência rápida**: [Cheatsheet](docs/CHEATSHEET.md)

---

**FIAP - Pós Tech Machine Learning Engineering**
