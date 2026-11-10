# Aula 04 - Deploy de Modelos

> **Repositório**: https://github.com/josenetoo/fiap-ml-aula04

## 🎯 Objetivo

Transformar o modelo treinado (Aula 03) em **artefato deployável**: script batch para predições em lote, **API REST com Flask** para tempo real, tudo containerizado com **Docker** e orquestrado com **Docker Compose**.

## 📹 Vídeos desta Aula

| Vídeo | Tema | O que você vai fazer |
|-------|------|---------------------|
| 01 | Padrões de Deploy: Batch vs Tempo Real | Implementar predição em batch e comparar com tempo real |
| 02 | Construção de API REST com Flask | API com `/health` e `/predict`, validação de input e testes |
| 03 | Ambientes Reprodutíveis com Docker | Dockerfile, build, run, Docker Compose, multi-stage |

## 🏗️ Pipeline desta Aula

```
┌───────┐         ┌──────────┐     ┌─────────┐
│ Train │────────▶│  model   │────▶│  Batch  │ (CSV → CSV)
│       │         │  .pkl    │     └─────────┘
└───────┘         │          │     ┌─────────┐    ┌────────┐
                  │          │────▶│ Flask   │───▶│ Docker │ (HTTP)
                  └──────────┘     │ API     │    │        │
                                   └─────────┘    └────────┘
```

Detalhes em [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## � Estrutura do Repositório

```
.
├── .gitignore
├── .dockerignore
├── README.md
├── requirements.txt
├── pytest.ini
├── Dockerfile                 # Imagem padrão
├── Dockerfile.slim            # Multi-stage (menor)
├── docker-compose.yml         # API + MLflow
├── docs/
│   ├── ARCHITECTURE.md
│   ├── CHEATSHEET.md
│   ├── HANDS-ON-04-01.md      # Batch
│   ├── HANDS-ON-04-02.md      # Flask
│   └── HANDS-ON-04-03.md      # Docker
├── src/
│   ├── train.py
│   ├── batch_predict.py
│   └── api/app.py
└── tests/
    ├── test_api.py
    └── test_batch.py
```

## Pré-requisitos

| Requisito | Como verificar |
|-----------|----------------|
| Aula 03 concluída | Pipeline da Aula 03 rodando |
| Python 3.9+ | `python3 --version` |
| Docker Desktop | `docker --version` e `docker ps` |
| Porta 5001 livre | `lsof -i :5001` (Mac/Linux) |

> ⚠️ Usamos **porta 5001** (não 5000) por causa do AirPlay no macOS Sonoma+.

## 🚀 Como Usar

1. **Fork** e clone este repositório
2. `pip install -r requirements.txt`
3. `python src/train.py` (treina e salva modelo)
4. Siga os hands-on em `docs/HANDS-ON-04-*.md`

## 📚 Documentação

| Vídeo | Hands-on |
|-------|----------|
| 01 - Padrões de Deploy: Batch vs Tempo Real | [HANDS-ON-04-01.md](docs/HANDS-ON-04-01.md) |
| 02 - Construção de API REST com Flask | [HANDS-ON-04-02.md](docs/HANDS-ON-04-02.md) |
| 03 - Ambientes Reprodutíveis com Docker | [HANDS-ON-04-03.md](docs/HANDS-ON-04-03.md) |

**Referência rápida**: [Cheatsheet](docs/CHEATSHEET.md)

---

**FIAP - Pós Tech Machine Learning Engineering**
