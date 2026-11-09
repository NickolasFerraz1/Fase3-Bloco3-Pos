# 🎬 Vídeo 2.2 - Transformação e Seleção de Features

**Aula**: 2 - Ingestão e Feature Engineering  
**Vídeo**: 2.2  
**Temas**: Normalização; Encoding; Pipeline sklearn; ColumnTransformer

---

## 🚀 Sobre Este Vídeo

> **"Modelo só entende número."** Hoje você transforma texto e categorias em features numéricas usando `Pipeline` do sklearn.

### O que você vai fazer:

| Etapa | Descrição |
|-------|-----------|
| **StandardScaler** | Normalizar features numéricas |
| **OneHotEncoder** | Encoding de categóricas |
| **Pipeline** | Encadear transformações |
| **ColumnTransformer** | Aplicar transformação por coluna |

### Pré-requisitos

| Requisito | Como verificar |
|-----------|----------------|
| Aula 2.1 concluída | `src/data/validate.py` existe |
| Pytest passando | `pytest tests/` retorna verde |

---

## 📚 Parte 1: Por Que Transformar?

### Passo 1: O Problema da Escala

**Cenário:**

| Cliente | Idade | Salário |
|---------|-------|---------|
| A | 25 | 3000 |
| B | 45 | 8000 |
| C | 35 | 5000 |

O modelo vê `salário` (milhares) como **muito maior** que `idade` (dezenas). Algoritmos que usam distância (KNN, SVM, redes) ficam enviesados.

**Solução:** colocar todas features na mesma escala (~0 a 1, ou média 0 / desvio 1).

> 💡 **Ponto-chave**: Árvores (RandomForest, XGBoost) **não precisam** de normalização. Modelos lineares e neurais **precisam**.

---

### Passo 2: O Problema das Categorias

**Cenário:**

| Cliente | Cidade |
|---------|--------|
| A | "SP" |
| B | "RJ" |
| C | "MG" |

Modelo não entende texto. Soluções:

- **OneHotEncoder**: vira 3 colunas binárias `is_SP`, `is_RJ`, `is_MG`.
- **LabelEncoder**: vira inteiro (`SP=0, RJ=1, MG=2`) — ⚠️ pode induzir ordem falsa.

---

## 🛠️ Parte 2: Normalização

### Passo 3: StandardScaler

**Criar `src/features/transform.py`:**

> 📄 **O que é este arquivo**: a "cozinha" das features — onde o dado é preparado no formato que o modelo entende.
> **O que ele faz (por enquanto)**: cria um `StandardScaler` que coloca os números na **mesma escala**
> (média ~0, desvio ~1), para que nenhuma coluna "domine" as outras só por ter valores maiores.

```python
"""Transformações de features."""
import logging
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def fit_scaler(X_train):
    """Treina scaler apenas no treino."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    logger.info(f"Scaler treinado. Médias: {scaler.mean_[:3].round(2)}")
    return scaler


def transform(scaler, X):
    """Aplica scaler em qualquer conjunto."""
    return scaler.transform(X)
```

> ⚠️ **CRÍTICO**: `fit` SÓ no treino. Aplicar `transform` em treino E teste. Senão tem **data leakage**.

---

### Passo 4: Demonstrar Efeito

**Executar no terminal (não precisa de `cd src`, só usa sklearn):**

```bash
python -c "import numpy as np
from sklearn.preprocessing import StandardScaler
X = np.array([[25, 3000], [45, 8000], [35, 5000]])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print('Antes:'); print(X)
print('Depois:'); print(X_scaled.round(2))
print(f'Médias: {X_scaled.mean(axis=0).round(2)}')
print(f'Desvios: {X_scaled.std(axis=0).round(2)}')"
```

**Resultado esperado:**
```
Antes:
[[  25 3000]
 [  45 8000]
 [  35 5000]]

Depois:
[[-1.22 -1.16]
 [ 1.22  1.27]
 [ 0.   -0.11]]

Médias: [0. 0.]
Desvios: [1. 1.]
```

> 💡 **Ponto-chave**: Após `StandardScaler`, média ≈ 0 e desvio ≈ 1 em todas colunas.

✅ Normalização funcionando.

---

## 🔤 Parte 3: Encoding de Categorias

### Passo 5: OneHotEncoder

**No terminal (não precisa de `cd src`, só usa sklearn):**

```bash
python -c "import pandas as pd
from sklearn.preprocessing import OneHotEncoder
df = pd.DataFrame({'cidade': ['SP', 'RJ', 'MG', 'SP', 'RJ']})
encoder = OneHotEncoder(sparse_output=False)
encoded = encoder.fit_transform(df[['cidade']])
print('Categorias:', encoder.categories_[0])
print('Encoded:'); print(encoded)"
```

**Resultado esperado:**
```
Categorias: ['MG' 'RJ' 'SP']
Encoded:
 [[0. 0. 1.]   # SP
  [0. 1. 0.]   # RJ
  [1. 0. 0.]   # MG
  [0. 0. 1.]   # SP
  [0. 1. 0.]]  # RJ
```

✅ Categórica virou 3 colunas numéricas.

---

## 🔗 Parte 4: Pipeline do sklearn

### Passo 6: Encadear Transformações

**Atualizar `src/features/transform.py`:**

> 📄 **O que estamos adicionando**: a função `build_numeric_pipeline`.
> **O que ela faz**: encadeia dois passos numa sequência fixa — primeiro **preenche valores faltantes**
> (imputação) e depois **normaliza** (scaler). Um `Pipeline` garante que a ordem seja sempre a mesma
> e que tudo vire **um único objeto** fácil de reusar e salvar.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def build_numeric_pipeline():
    """Pipeline: imputação -> normalização."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
```

**Por que pipeline?**

| Sem Pipeline | Com Pipeline |
|--------------|--------------|
| Chamar `imputer.fit()`, `imputer.transform()`, `scaler.fit()`, `scaler.transform()` | Uma chamada: `pipe.fit_transform(X)` |
| Risco de esquecer ordem | Ordem definida na construção |
| Difícil testar | Pipeline é UM objeto |

> 💡 **Ponto-chave**: Pipeline é o `docker-compose` do pré-processamento — encadeia passos em ordem definida.

---

### Passo 7: ColumnTransformer (numérico + categórico)

**Adicionar a `src/features/transform.py`:**

> 📄 **O que estamos adicionando**: a função `build_full_preprocessor` (com `ColumnTransformer`).
> **O que ela faz**: aplica um tratamento **diferente para cada tipo de coluna** ao mesmo tempo —
> normaliza as numéricas e transforma as categóricas em número (one-hot). É a peça que junta tudo
> num pré-processador só.

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def build_full_preprocessor(numeric_cols, categorical_cols):
    """Aplica transformação diferente em cada tipo de coluna."""
    numeric_pipe = build_numeric_pipeline()
    categorical_pipe = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ])
```

---

### Passo 8: Testar Pipeline Completo

**Criar `tests/test_transform.py`:**

> 📄 **O que é este arquivo**: os testes do pré-processador.
> **O que ele faz**: monta uma tabelinha de exemplo (números + categoria) e confere que, depois de
> processada, o resultado tem o **número de colunas esperado** — prova que numéricas e categóricas
> foram tratadas corretamente.

```python
import pandas as pd
from features.transform import build_full_preprocessor


def test_pipeline_processa_dataframe_misto():
    df = pd.DataFrame({
        "idade": [25, 45, 35],
        "salario": [3000, 8000, 5000],
        "cidade": ["SP", "RJ", "MG"],
    })

    preprocessor = build_full_preprocessor(
        numeric_cols=["idade", "salario"],
        categorical_cols=["cidade"],
    )

    X = preprocessor.fit_transform(df)

    # 2 numéricas + 3 cidades = 5 colunas
    assert X.shape == (3, 5)
```

**Rodar:**

**Linux/Mac e Windows:**
```bash
pytest tests/test_transform.py -v
```

**Resultado esperado:**
```
tests/test_transform.py::test_pipeline_processa_dataframe_misto PASSED
=================== 1 passed in 0.32s ===================
```

✅ Pipeline misto funcionando.

---

## 🎯 Parte 5: Seleção de Features

### Passo 9: Importância das Features

**No terminal (não precisa de `cd src`, só usa sklearn):**

```bash
python -c "from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True)
feature_names = load_iris().feature_names
model = RandomForestClassifier(random_state=42).fit(X, y)
for name, importance in zip(feature_names, model.feature_importances_):
    print(f'{name:30s} {importance:.3f}')"
```

**Resultado esperado:**
```
sepal length (cm)              0.092
sepal width (cm)               0.025
petal length (cm)              0.434
petal width (cm)               0.449
```

> 💡 **Ponto-chave**: `petal length` e `petal width` somam ~88% da importância. As features de `sepal` quase não importam.

✅ Importância calculada.

---

## 🔧 Troubleshooting

| Erro | Causa | Solução |
|------|-------|---------|
| `ValueError: could not convert string to float` | Categórica não foi encodada | Aplicar OneHot antes do scaler |
| `sparse_output` não existe | sklearn < 1.2 | Atualizar: `pip install -U scikit-learn` |
| Pipeline lento | Dataset grande + Grid Search | Reduzir `n_estimators` ou usar `RandomizedSearch` |
| Data leakage | Usou `fit_transform` no teste | Usar SÓ `transform` (com scaler já treinado) |
| Categórica desconhecida no teste | Aparece label novo | `OneHotEncoder(handle_unknown="ignore")` |

---

**FIM DO VÍDEO 2.2** ✅
