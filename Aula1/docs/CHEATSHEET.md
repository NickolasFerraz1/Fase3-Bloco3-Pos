# Aula 01 - Cheatsheet

## Conceitos-Chave

```
MLOps = Machine Learning + DevOps + Data Engineering
```

### 5 Pilares do MLOps
1. **Automação** - Build, teste, deploy automáticos
2. **Versionamento** - Código + Dados + Modelo
3. **Reprodutibilidade** - Mesmo input = mesmo output
4. **Monitoramento** - Observar performance em produção
5. **Colaboração** - Times multidisciplinares

## Pipeline ML vs ETL

| Aspecto | ETL | Pipeline ML |
|---------|-----|-------------|
| Fluxo | Linear | Cíclico |
| Saída | Dados | Modelo |
| Validação | Schema | Métricas |
| Atualização | Dados mudam | Modelo degrada |

## Ferramentas Essenciais

```bash
# Versionamento de código
git init
git add .
git commit -m "Initial commit"

# Versionamento de dados (DVC)
dvc init
dvc add data/raw/dataset.csv
dvc push

# Ambiente virtual Python
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Instalar dependências
pip install -r requirements.txt

# Poetry (gerenciador moderno)
poetry init
poetry add pandas scikit-learn
poetry install
```

## Estrutura de Projeto ML

```
projeto-ml/
├── data/
│   ├── raw/              # Dados originais
│   └── processed/        # Dados processados
├── notebooks/            # Exploração
├── src/
│   ├── data/            # Scripts de ingestão
│   ├── features/        # Feature engineering
│   ├── models/          # Treinamento
│   └── utils/           # Utilitários
├── tests/               # Testes
├── models/              # Modelos salvos
├── .gitignore
├── requirements.txt
└── README.md
```

## Comandos Python Úteis

```python
# Fixar random seed (reprodutibilidade)
import random
import numpy as np
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Salvar modelo
import joblib
joblib.dump(modelo, 'modelo.pkl')

# Carregar modelo
modelo = joblib.load('modelo.pkl')

# Logging básico
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Treinamento iniciado")
```

## MLflow Básico

```python
import mlflow

# Iniciar experimento
mlflow.set_experiment("meu-experimento")

# Registrar run
with mlflow.start_run():
    # Parâmetros
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_param("epochs", 100)
    
    # Métricas
    mlflow.log_metric("accuracy", 0.95)
    mlflow.log_metric("loss", 0.05)
    
    # Modelo
    mlflow.sklearn.log_model(modelo, "model")
```

## Docker Básico

```bash
# Build imagem
docker build -t meu-modelo:v1 .

# Rodar container
docker run -p 5000:5000 meu-modelo:v1

# Listar containers
docker ps

# Parar container
docker stop <container_id>
```

## Git Workflow

```bash
# Criar branch
git checkout -b feature/novo-modelo

# Fazer mudanças
git add src/models/train.py
git commit -m "feat: adicionar modelo XGBoost"

# Push
git push origin feature/novo-modelo

# Merge (após PR aprovado)
git checkout main
git pull origin main
git merge feature/novo-modelo
```

## Boas Práticas

### ✅ Fazer
- Versionar código, dados e modelo
- Usar ambientes virtuais
- Fixar versões de dependências
- Escrever testes
- Documentar código
- Usar logging
- Monitorar em produção

### ❌ Evitar
- Notebook monolítico de 5000 linhas
- Hardcoded paths (`/Users/jose/dados.csv`)
- Credenciais no código
- `import *`
- Variáveis globais
- Código sem testes
- Deploy sem validação

## Troubleshooting

### Erro: "ModuleNotFoundError"
```bash
# Verificar ambiente ativo
which python

# Reinstalar dependências
pip install -r requirements.txt
```

### Erro: "Modelo não reproduz resultados"
```python
# Fixar todas as seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
```

### Erro: "Out of Memory"
```python
# Processar em batches
for batch in pd.read_csv('dados.csv', chunksize=10000):
    processar(batch)
```

## Referências Rápidas

- [MLOps Principles](https://ml-ops.org/content/mlops-principles)
- [DVC Documentation](https://dvc.org/doc)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Scikit-learn Best Practices](https://scikit-learn.org/stable/developers/develop.html)

---

**Última atualização**: Aula 01
