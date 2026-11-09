# 🎬 Vídeo 3.2 - Treino e Rastreabilidade com MLflow

**Aula**: 3 - Treinamento de Modelos e Validação  
**Vídeo**: 3.2  
**Temas**: MLflow Tracking; Experimentos; Model Registry; Comparação de runs

---

## 🚀 Sobre Este Vídeo

> **"MLflow é o Git pra experimentos de ML."** Cada treino vira um 'run' versionado, com parâmetros, métricas e modelo.

### O que você vai fazer:

| Etapa | Descrição |
|-------|-----------|
| **MLflow Tracking** | Logar params, metrics, artifacts |
| **UI do MLflow** | Visualizar e comparar runs |
| **Experiments** | Agrupar runs |
| **Model Registry** | Staging → Production |

### Pré-requisitos

| Requisito | Como verificar |
|-----------|----------------|
| Vídeo 3.1 concluído | `src/train/metrics.py` existe |
| MLflow instalado | `mlflow --version` |
| Porta 5001 livre | `lsof -i :5001` (Mac/Linux) |

---

## 📚 Parte 1: O Problema Que MLflow Resolve

### Passo 1: O Caos Sem MLflow

**Cenário comum:**

```
treino_v1.py           → accuracy = 0.85
treino_v2_n_est100.py  → accuracy = 0.87
treino_v3_max_depth5.py → accuracy = 0.91
treino_v4_TESTAR.py    → accuracy = 0.93  ← qual era esse mesmo?
```

| Problema | Impacto |
|----------|---------|
| Qual modelo está em produção? | Ninguém sabe |
| Como reproduzir? | Impossível |
| Qual era o melhor hiperparâmetro? | Tem que rodar tudo de novo |

> 💡 **Ponto-chave**: MLflow tem cara de **Jenkins/Grafana** — UI web, histórico de runs, comparação visual. Sua zona de conforto DevOps.

---

### Passo 2: Instalar e Verificar o MLflow

Se você seguiu o Vídeo 3.1, o MLflow já veio no `pip install -r requirements.txt`. Confirme:

**Linux/Mac e Windows:**
```bash
mlflow --version
```

**Resultado esperado:**
```
mlflow, version 2.8.1
```

Se aparecer `command not found` / `não reconhecido`, o ambiente virtual não está ativo ou o MLflow não foi instalado:

**Linux/Mac:**
```bash
source venv/bin/activate
pip install mlflow==2.8.1
```

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
pip install mlflow==2.8.1
```

> 🧠 **Por que verificar antes:** o `mlflow` é um comando de terminal (CLI) que só existe se o pacote estiver instalado no ambiente ativo. Conferir a versão primeiro evita o erro clássico de "subir a UI" num terminal onde o `venv` não está ativado. **Analogia:** é o mesmo que rodar `docker --version` antes de `docker run`.

> ⚠️ **Erro `ModuleNotFoundError: pkg_resources`?** No Python 3.11+ o `setuptools` não vem mais no venv por padrão, e versões 81+ removeram o `pkg_resources` que o MLflow usa. Solução: `pip install "setuptools<81"` (já incluído no `requirements.txt` desta aula). Pode aparecer um `UserWarning` de depreciação — é só aviso, ignore.

✅ MLflow instalado e disponível.

---

### Passo 3: Subir o MLflow Local

> ⚠️ **Porta no macOS:** a porta 5000 é ocupada pelo serviço **AirPlay Receiver** do macOS. Por isso usamos a **5001** como padrão aqui. No Linux/Windows a 5000 funciona normalmente, mas vamos manter 5001 pra todo mundo seguir igual.

**Linux/Mac:**
```bash
mlflow ui --host 0.0.0.0 --port 5001
```

**Windows (PowerShell):**
```powershell
mlflow ui --host 0.0.0.0 --port 5001
```

**Resultado esperado:**
```
[2026-06-26 10:00:00 +0000] [12345] [INFO] Starting gunicorn 21.2.0
[2026-06-26 10:00:00 +0000] [12345] [INFO] Listening at: http://0.0.0.0:5001
```

**Abrir no browser**: http://localhost:5001

✅ UI do MLflow rodando.

> 🧠 **O que é o `mlflow ui`:** sobe um servidor web local (igual subir um Grafana ou um Jenkins) que mostra o histórico de todos os treinos. `--host 0.0.0.0` = aceita conexões de fora; `--port 5001` = porta de acesso.

> ⚠️ **IMPORTANTE**: Deixe esse terminal aberto. Abra **outro** terminal para os próximos passos.

---

## 🛠️ Parte 2: Logar Primeiro Run

### Passo 4: Treino com Logging

**Criar `src/train/train_mlflow.py`:**

```python
"""Treino com rastreamento via MLflow."""
import logging
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier

from train.split import get_train_test_split
from train.metrics import evaluate

logger = logging.getLogger(__name__)


def train_run(n_estimators=100, max_depth=None, experiment_name="iris-classification"):
    """Treina e loga no MLflow."""
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        # 1. Dados
        X_train, X_test, y_train, y_test = get_train_test_split()

        # 2. Treinar
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
        )
        model.fit(X_train, y_train)

        # 3. Avaliar
        metrics, cm = evaluate(model, X_test, y_test)

        # 4. Logar TUDO
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("model_type", "RandomForest")

        for name, value in metrics.items():
            mlflow.log_metric(name, value)

        mlflow.sklearn.log_model(model, "model")

        logger.info(f"✅ Run logado. Accuracy: {metrics['accuracy']:.3f}")
        return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_run()
```

---

### Passo 5: Rodar Primeiro Treino

**Linux/Mac e Windows:**
```bash
PYTHONPATH=src python src/train/train_mlflow.py
```

**Resultado esperado:**
```
INFO: Treino: 120 | Teste: 30
INFO: Métricas: {'accuracy': 0.967, ...}
INFO: ✅ Run logado. Accuracy: 0.967
```

**Atualizar a UI do MLflow** (F5): você verá o experimento `iris-classification` com 1 run.

✅ Primeiro run logado.

> 🧠 **O que o código logou:** cada `mlflow.log_param` grava uma configuração de entrada (ex: `n_estimators`) e cada `mlflow.log_metric` grava um resultado (ex: `accuracy`). O `log_model` salva o modelo treinado como artefato. **Analogia:** é exatamente um build no CI — params são as variáveis de build, metrics são os resultados dos testes, e o modelo é o artefato publicado. Cada `run` é um build com hash próprio.

---

## 🔬 Parte 3: Comparar Experimentos

### Passo 6: Rodar Múltiplos Treinos

**Linux/Mac e Windows:**
```bash
PYTHONPATH=src python -c "
from train.train_mlflow import train_run
for n in [10, 50, 100, 200]:
    for d in [3, 5, None]:
        train_run(n_estimators=n, max_depth=d)
"
```

**Resultado esperado:** 12 runs aparecem na UI.

---

### Passo 7: Comparar na UI

**Na UI do MLflow (http://localhost:5001):**

1. Clique no experimento `iris-classification`
2. Selecione todos os runs (checkbox)
3. Clique em **Compare**
4. Veja:
   - Tabela comparativa de params e métricas
   - Gráfico de coordenadas paralelas
   - Scatter plot

> 💡 **Ponto-chave**: Em segundos você identifica visualmente o melhor modelo.

✅ Comparação visual feita.

---

## 📦 Parte 4: Model Registry

### Passo 8: O Que É o Registry?

**Analogia DevOps direta:**

| Container Registry (ECR/Docker Hub) | MLflow Model Registry |
|-------------------------------------|----------------------|
| Tags: `dev`, `staging`, `prod` | Stages: `None`, `Staging`, `Production`, `Archived` |
| Versão semântica | Versão automática (1, 2, 3...) |
| `docker pull image:prod` | `mlflow.sklearn.load_model("models:/iris/Production")` |

---

### Passo 9: Registrar Modelo

**Linux/Mac e Windows:**
```bash
PYTHONPATH=src python -c "
import mlflow

# Pegar o run com melhor accuracy
client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name('iris-classification')
runs = client.search_runs(
    experiment.experiment_id,
    order_by=['metrics.accuracy DESC'],
    max_results=1,
)
best_run = runs[0]
print(f'Melhor run: {best_run.info.run_id} | accuracy={best_run.data.metrics[\"accuracy\"]:.3f}')

# Registrar como 'iris-classifier'
model_uri = f'runs:/{best_run.info.run_id}/model'
result = mlflow.register_model(model_uri, 'iris-classifier')
print(f'Registrado: {result.name} v{result.version}')
"
```

**Resultado esperado:**
```
Melhor run: abc123... | accuracy=0.967
Successfully registered model 'iris-classifier'
Registrado: iris-classifier v1
```

---

### Passo 10: Promover para Production

**Linux/Mac e Windows:**
```bash
PYTHONPATH=src python -c "
from mlflow.tracking import MlflowClient

client = MlflowClient()
client.transition_model_version_stage(
    name='iris-classifier',
    version=1,
    stage='Production',
)
print('✅ Modelo promovido para Production')
"
```

**Resultado esperado:**
```
✅ Modelo promovido para Production
```

**Na UI**: clique em **Models** (topo) → `iris-classifier` → versão 1 está em `Production`.

✅ Modelo em produção (no Registry).

> 🧠 **Stage = tag de ambiente.** Promover pra `Production` não move arquivo nenhum — só muda um rótulo apontando qual versão é a "oficial". **Analogia direta:** é idêntico a re-taggear uma imagem Docker de `staging` pra `prod` no registry. A API de inferência sempre pede "me dá a versão `Production`" e não precisa saber qual número é. Rollback = só apontar o stage pra versão anterior.

---

### Passo 11: Carregar Modelo de Produção

**Linux/Mac e Windows:**
```bash
PYTHONPATH=src python -c "
import mlflow
import numpy as np

model = mlflow.sklearn.load_model('models:/iris-classifier/Production')

# Predição
X_novo = np.array([[5.1, 3.5, 1.4, 0.2]])
pred = model.predict(X_novo)
print(f'Predição: classe {pred[0]}')
"
```

**Resultado esperado:**
```
Predição: classe 0
```

> 💡 **Ponto-chave**: A API de inferência (Aula 04) vai carregar o modelo **dessa forma**. Trocar de modelo = só mudar o stage no Registry.

✅ Modelo carregado e usado.

---

## 🔧 Troubleshooting

| Erro | Causa | Solução |
|------|-------|---------|
| `Connection refused` ao logar | MLflow UI não está rodando | Iniciar `mlflow ui --port 5001` em outro terminal |
| Porta 5000 ocupada | macOS usa 5000 (AirPlay) | Usar `--port 5001` (padrão deste hands-on) |
| `ModuleNotFoundError: mlflow` | Não instalado ou venv inativo | Ativar venv e `pip install mlflow==2.8.1` |
| `ModuleNotFoundError: pkg_resources` | `setuptools` 81+ removeu o `pkg_resources` que o mlflow usa | `pip install "setuptools<81"` (já no `requirements.txt`) |
| Experimento não aparece | Pasta `mlruns/` em local diferente | Rodar sempre da mesma pasta |
| Modelo não carrega via URI | Stage errado | Verificar na UI o stage exato (case-sensitive) |

---

**FIM DO VÍDEO 3.2** ✅
