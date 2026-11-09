# 🎬 Vídeo 3.1 - Generalização, Overfitting e Métricas

**Aula**: 3 - Treinamento de Modelos e Validação  
**Vídeo**: 3.1  
**Temas**: Train/Test Split; Overfitting; Métricas; Matriz de Confusão

---

## 🚀 Sobre Este Vídeo

> **"Modelo com 100% no treino e 60% em produção. O que aconteceu? Decorou em vez de aprender."**

### O que você vai fazer:

| Etapa | Descrição |
|-------|-----------|
| **Train/Test Split** | Separar dados de treino e teste |
| **Overfitting** | Detectar e demonstrar |
| **Métricas** | Accuracy, Precision, Recall, F1 |
| **Matriz de Confusão** | Visualizar erros do modelo |

### Pré-requisitos

| Requisito | Como verificar |
|-----------|----------------|
| Aula 02 concluída | `iris_features.csv` foi gerado |
| Python 3.11 | `python3.11 --version` |
| scikit-learn | `python -c "import sklearn; print(sklearn.__version__)"` |

---

## 📚 Parte 1: Por Que Separar Treino e Teste?

### Passo 1: A Analogia do Simulado

| Situação | DevOps | ML |
|----------|--------|-----|
| Validar em DEV apenas | Bug aparece em PROD | Accuracy 100% no treino, 60% em produção |
| Validar em STAGING separada | Confiança no deploy | Accuracy estável em dados novos |

> 💡 **Ponto-chave**: Treinar e avaliar no **mesmo dado** = mesma situação que QA testar no ambiente do dev. Não funciona.

---

### Passo 2: Setup do Projeto

**Linux/Mac:**
```bash
mkdir -p ~/fiap-mlops/aula03
cd ~/fiap-mlops/aula03
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell):**
```powershell
New-Item -ItemType Directory -Path "$HOME\fiap-mlops\aula03" -Force
cd "$HOME\fiap-mlops\aula03"
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> 🐍 **Por que Python 3.11?** As versões das bibliotecas estão "pinadas" (travadas) no `requirements.txt` em versões testadas — scikit-learn 1.3.0, numpy 1.24.3 etc. Elas não compilam no Python 3.12+. O `3.11` é a faixa garantida. Pense nisso como fixar a versão de uma imagem base no Dockerfile em vez de usar `latest`.
>
> 🔌 **`venv` + `activate`**: o `venv` cria um ambiente Python isolado só pra este projeto (igual um container isola dependências). O `activate` "entra" nesse ambiente. Tudo que você instalar fica preso aqui, sem sujar o Python do sistema.

**Resultado esperado:**
```
Successfully installed scikit-learn-1.3.0 pandas-2.0.3 mlflow-2.8.1 ...
```

✅ Ambiente pronto.

---

### Passo 3: Fazer o Split

**Criar `src/train/split.py`:**

```python
"""Divisão treino/teste."""
import logging
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def get_train_test_split(test_size=0.2, random_state=42):
    """Retorna (X_train, X_test, y_train, y_test)."""
    X, y = load_iris(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,  # Mantém proporção de classes
    )

    logger.info(f"Treino: {len(X_train)} | Teste: {len(X_test)}")
    return X_train, X_test, y_train, y_test
```

> ⚠️ **IMPORTANTE**: `stratify=y` garante que a proporção de classes seja a mesma em treino e teste. **Crítico para datasets desbalanceados.**

---

### Passo 4: Validar o Split

**Executar:**

**Linux/Mac:**
```bash
mkdir -p src/train
# (criar arquivo acima)
python -c "
from sys import path; path.insert(0, 'src')
from train.split import get_train_test_split
import numpy as np
X_tr, X_te, y_tr, y_te = get_train_test_split()
print(f'Treino: {len(X_tr)}')
print(f'Teste: {len(X_te)}')
print(f'Classes treino: {np.bincount(y_tr)}')
print(f'Classes teste: {np.bincount(y_te)}')
"
```

**Windows (PowerShell):**
```powershell
New-Item -ItemType Directory -Path src\train -Force
$env:PYTHONPATH = "src"
python -c "from train.split import get_train_test_split; import numpy as np; X_tr,X_te,y_tr,y_te = get_train_test_split(); print(f'Treino:{len(X_tr)}'); print(f'Classes treino:{np.bincount(y_tr)}')"
```

**Resultado esperado:**
```
Treino: 120
Teste: 30
Classes treino: [40 40 40]
Classes teste: [10 10 10]
```

✅ Split balanceado.

> 🧠 **O que aconteceu aqui:** separamos os dados em dois grupos — 120 exemplos pra o modelo aprender (treino) e 30 que ele NUNCA vê durante o aprendizado (teste). O `[40 40 40]` mostra que as 3 espécies de flor ficaram igualmente distribuídas (isso é o `stratify`). **Analogia:** é como separar uma base de testes que o código nunca viu em desenvolvimento — se funcionar nela, você confia que generaliza.

---

## 🎯 Parte 2: Detectar Overfitting

### Passo 5: Modelo Que Decora

**Criar `src/train/demo_overfit.py`:**

```python
"""Demo: overfitting vs modelo balanceado."""
from sklearn.tree import DecisionTreeClassifier
from train.split import get_train_test_split


def comparar_profundidades():
    X_train, X_test, y_train, y_test = get_train_test_split()

    print(f"{'max_depth':<12} {'train':>8} {'test':>8} {'gap':>8}")
    print("-" * 40)

    for depth in [1, 3, 5, None]:  # None = sem limite (overfitting!)
        model = DecisionTreeClassifier(max_depth=depth, random_state=42)
        model.fit(X_train, y_train)

        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        gap = train_score - test_score

        label = "None" if depth is None else str(depth)
        print(f"{label:<12} {train_score:>8.3f} {test_score:>8.3f} {gap:>8.3f}")


if __name__ == "__main__":
    comparar_profundidades()
```

**Executar:**

**Linux/Mac:**
```bash
PYTHONPATH=src python src/train/demo_overfit.py
```

**Windows (PowerShell):**
```powershell
$env:PYTHONPATH = "src"
python src\train\demo_overfit.py
```

**Resultado esperado:**
```
max_depth      train     test      gap
----------------------------------------
1              0.667    0.667    0.000   ← underfitting
3              0.975    0.967    0.008   ← bom!
5              1.000    0.967    0.033   ← começa a decorar
None           1.000    0.933    0.067   ← OVERFITTING
```

> 💡 **Leitura**: `gap` grande = modelo decorou. `train` e `test` baixos = modelo é burro (underfitting).

✅ Overfitting demonstrado.

> 🧠 **A regra de ouro:** `gap` = nota no treino − nota no teste. Quanto maior o gap, mais o modelo "decorou" e vai falhar com dado novo. **Analogia DevOps:** funciona 100% na máquina do dev (`train`) e quebra em produção (`test`) — o famoso "na minha máquina funciona". O `max_depth` controla a complexidade do modelo: alto demais = decora.

---

## 📊 Parte 3: Métricas Além de Accuracy

### Passo 6: Por Que Accuracy Engana?

**Cenário**: Sistema de detecção de fraude (1% das transações são fraude).

| Modelo | Predição | Accuracy |
|--------|----------|----------|
| "Sempre digo: NÃO é fraude" | Todas negativo | **99%** ⚠️ |
| Modelo de verdade | Acerta 90% | 90% |

O modelo burro tem accuracy **maior**! Por isso precisamos de **precision** e **recall**.

---

### Passo 7: Precision e Recall Explicados Simples

**Cenário**: detector de fraude em transações de cartão.

| Métrica | Pergunta | Importância |
|---------|----------|-------------|
| **Precision** | "Dos que **eu disse** que era fraude, quantos eram?" | Evita bloquear compra legítima |
| **Recall** | "Dos que **eram** fraude, quantos eu peguei?" | Não deixa passar fraude real |
| **F1-Score** | Média harmônica de precision e recall | Equilíbrio |

> ⚠️ **Trade-off**: Aumentar recall geralmente diminui precision e vice-versa.

---

### Passo 8: Calcular as Métricas

**Criar `src/train/metrics.py`:**

```python
"""Cálculo de métricas de classificação."""
import logging
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


def evaluate(model, X_test, y_test):
    """Retorna dict com todas as métricas."""
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1": f1_score(y_test, y_pred, average="weighted"),
    }

    logger.info(f"Métricas: {metrics}")
    logger.info("\n" + classification_report(y_test, y_pred))

    return metrics, confusion_matrix(y_test, y_pred)
```

---

### Passo 9: Avaliar um Modelo

**Executar:**

**Linux/Mac e Windows:**
```bash
PYTHONPATH=src python -c "
import logging
logging.basicConfig(level=logging.INFO)

from sklearn.ensemble import RandomForestClassifier
from train.split import get_train_test_split
from train.metrics import evaluate

X_tr, X_te, y_tr, y_te = get_train_test_split()
model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_tr, y_tr)
metrics, cm = evaluate(model, X_te, y_te)
print('Métricas:', metrics)
print('Matriz de Confusão:')
print(cm)
"
```

> 🧠 **Por que precisa do `logging.basicConfig`?** A função `evaluate()` mostra as métricas via `logger.info()`. Sem configurar o logging, o Python esconde mensagens de nível INFO por padrão — por isso só a matriz (que usa `print`) aparecia. Essa linha "liga" a exibição dos logs. O `print('Métricas:', metrics)` garante que elas apareçam de qualquer jeito.

**Resultado esperado:**
```
Métricas: {'accuracy': 0.967, 'precision': 0.967, 'recall': 0.967, 'f1': 0.967}

              precision    recall  f1-score   support
           0       1.00      1.00      1.00        10
           1       0.91      1.00      0.95        10
           2       1.00      0.90      0.95        10

    accuracy                           0.97        30

Matriz de Confusão:
[[10  0  0]
 [ 0 10  0]
 [ 0  1  9]]
```

✅ Modelo avaliado em múltiplas métricas.

> 🧠 **As 4 métricas em uma frase cada:**
> - **Accuracy**: % geral de acertos. Engana quando as classes são desbalanceadas.
> - **Precision**: dos que o modelo DISSE serem da classe X, quantos eram mesmo? (evita alarme falso)
> - **Recall**: dos que ERAM da classe X, quantos o modelo pegou? (evita deixar passar)
> - **F1**: equilíbrio entre precision e recall, num número só.
>
> **Analogia DevOps:** Precision alta = poucos alertas falsos no monitoramento (não acorda o on-call à toa). Recall alto = não deixa passar incidente real. Quase sempre há trade-off entre os dois.

---

## 🧩 Parte 4: Lendo a Matriz de Confusão

### Passo 10: Interpretação

```
            Predito 0  Predito 1  Predito 2
Real 0  [    10         0          0     ]
Real 1  [     0        10          0     ]
Real 2  [     0         1          9     ]
                       ↑
                Classe 2 confundida com 1
```

| Posição | Significado |
|---------|-------------|
| **Diagonal** | Acertos (quanto mais, melhor) |
| **Fora da diagonal** | Erros |
| Linha `i`, Coluna `j` | "Era classe `i`, modelo previu `j`" |

> 💡 **Ponto-chave**: A matriz mostra **onde** o modelo erra, não só quanto.

---

## 🔧 Troubleshooting

| Erro | Causa | Solução |
|------|-------|---------|
| `ModuleNotFoundError: train` | `pythonpath` não configurado | Setar `PYTHONPATH=src` ou ajustar `pytest.ini` |
| Accuracy 100% no teste | Vazamento de dados | Verificar se teste foi usado no fit |
| `ValueError: stratify` | Classe com 1 exemplo só | Remover classe ou aumentar `test_size` |
| Métricas diferentes a cada execução | Sem `random_state` | Fixar `random_state=42` no `train_test_split` E no modelo |
| `UndefinedMetricWarning` | Classe sem predições | Usar `zero_division=0` em `precision_score` |

---

**FIM DO VÍDEO 3.1** ✅
