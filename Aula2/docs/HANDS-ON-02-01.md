# 🎬 Vídeo 2.1 - Qualidade de Dados e Validação

**Aula**: 2 - Ingestão e Feature Engineering  
**Vídeo**: 2.1  
**Temas**: Garbage In, Garbage Out; Validação de schema; Estatísticas; Outliers

---

## 🚀 Sobre Este Vídeo

> **"Modelo bom com dado ruim = previsão errada."** Hoje você garante que o input do pipeline está limpo.

### O que você vai fazer:

| Etapa | Descrição |
|-------|-----------|
| **Schema** | Validar colunas e tipos esperados |
| **Estatísticas** | Inspecionar média, desvio, min/max |
| **Outliers** | Detectar valores extremos com IQR |
| **NaN** | Tratar valores faltantes |

### Pré-requisitos

| Requisito | Como verificar |
|-----------|----------------|
| Aula 01 concluída | `~/fiap-mlops/aula01/src/pipeline.py` existe |
| venv ativado | Prompt mostra `(venv)` |
| Pandas instalado | `python -c "import pandas; print(pandas.__version__)"` |

---

## 📚 Parte 1: Por que Validar Dados?

### Passo 1: O Conceito GIGO

**GIGO** = *Garbage In, Garbage Out*

> 💡 **Ponto-chave**: Se você alimenta o modelo com dados ruins, vai obter predições ruins. **Não importa o algoritmo.**

**Problemas comuns nos dados:**

| Problema | Impacto |
|----------|---------|
| **Valores faltantes** | Modelo não treina ou faz suposições erradas |
| **Outliers** | Enviesam estatísticas e modelo |
| **Tipos errados** | `"25"` (string) em vez de `25` (int) |
| **Escalas diferentes** | Feature de 0-100 vs 0-1.000.000 |
| **Duplicados** | Overfitting em exemplos repetidos |

---

### Passo 2: Setup do Projeto Aula 02

**Linux/Mac:**
```bash
mkdir -p ~/fiap-mlops/aula02
cd ~/fiap-mlops/aula02
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell):**
```powershell
New-Item -ItemType Directory -Path "$HOME\fiap-mlops\aula02" -Force
cd "$HOME\fiap-mlops\aula02"
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Resultado esperado:**
```
Successfully installed scikit-learn-1.x pandas-2.x numpy-1.x ...
```

✅ Ambiente pronto.

---

## 🔍 Parte 2: Validação de Schema

### Passo 3: Criar Dataset de Exemplo

**Criar `src/data/load.py`:**

> 📄 **O que é este arquivo**: o ponto de entrada dos dados — a "porta" por onde o dado chega no pipeline.
> **O que ele faz**: carrega o dataset Iris e o entrega como uma **tabela** (DataFrame do pandas)
> com as colunas já nomeadas, pronto para os próximos passos validarem e transformarem.

```python
"""Carrega Iris como DataFrame."""
import pandas as pd
from sklearn.datasets import load_iris

def load_iris_df():
    iris = load_iris(as_frame=True)
    df = iris.frame
    df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "target"]
    return df

if __name__ == "__main__":
    df = load_iris_df()
    print(df.head())
    print(f"\nShape: {df.shape}")
```

**Executar:**

**Linux/Mac:**
```bash
mkdir -p src/data
# (criar o arquivo acima)
python src/data/load.py
```

**Windows (PowerShell):**
```powershell
New-Item -ItemType Directory -Path src\data -Force
python src\data\load.py
```

**Resultado esperado:**
```
   sepal_length  sepal_width  petal_length  petal_width  target
0           5.1          3.5           1.4          0.2       0
1           4.9          3.0           1.4          0.2       0
...

Shape: (150, 5)
```

✅ Dataset carregado como DataFrame.

---

### Passo 4: Validar Schema

**Criar `src/data/validate.py`:**

> 📄 **O que é este arquivo**: a "portaria" de qualidade dos dados.
> **O que ele faz**: a função `validate_schema` confere se a tabela tem as **colunas certas**
> e os **tipos certos**. Se faltar coluna ou o tipo estiver errado, ele **barra com um erro**
> de propósito — melhor parar aqui do que treinar um modelo com dado quebrado.

```python
"""Validação de schema do dataset."""
import logging
import pandas as pd

logger = logging.getLogger(__name__)

EXPECTED_SCHEMA = {
    "sepal_length": "float64",
    "sepal_width": "float64",
    "petal_length": "float64",
    "petal_width": "float64",
    "target": "int64",
}


def validate_schema(df: pd.DataFrame) -> bool:
    """Valida colunas e tipos."""
    logger.info("Validando schema...")

    # 1. Colunas obrigatórias
    missing = set(EXPECTED_SCHEMA) - set(df.columns)
    assert not missing, f"Colunas faltando: {missing}"

    # 2. Tipos
    for col, expected_type in EXPECTED_SCHEMA.items():
        actual_type = str(df[col].dtype)
        assert actual_type == expected_type, (
            f"Tipo errado em '{col}': esperado {expected_type}, obtido {actual_type}"
        )

    logger.info(f"✅ Schema válido: {len(df)} linhas x {len(df.columns)} colunas")
    return True
```

---

### Passo 5: Testar Validação

**Criar e executar `tests/test_validate.py`:**

> 📄 **O que é este arquivo**: os testes automáticos da validação.
> **O que ele faz**: prova que a validação funciona nos dois sentidos — aceita um dado bom
> e **rejeita** um dado ruim (coluna faltando ou tipo errado). É a nossa rede de segurança:
> se alguém quebrar a validação no futuro, o teste avisa na hora.

```python
import pytest
import pandas as pd
from data.load import load_iris_df
from data.validate import validate_schema


def test_schema_iris_e_valido():
    df = load_iris_df()
    assert validate_schema(df) is True


def test_schema_coluna_faltando_falha():
    df = load_iris_df().drop(columns=["target"])
    with pytest.raises(AssertionError, match="Colunas faltando"):
        validate_schema(df)


def test_schema_tipo_errado_falha():
    df = load_iris_df()
    df["target"] = df["target"].astype(str)
    with pytest.raises(AssertionError, match="Tipo errado"):
        validate_schema(df)
```

**Rodar:**

**Linux/Mac e Windows:**
```bash
pytest tests/ -v
```

**Resultado esperado:**
```
tests/test_validate.py::test_schema_iris_e_valido PASSED
tests/test_validate.py::test_schema_coluna_faltando_falha PASSED
tests/test_validate.py::test_schema_tipo_errado_falha PASSED
============== 3 passed in 0.45s ==============
```

✅ Schema validado e testado.

---

## 📊 Parte 3: Estatísticas Descritivas

### Passo 6: Inspecionar Dados

**No terminal (a partir da raiz do projeto):**

> 💡 Os imports `from data...` vivem em `src/`. O `PYTHONPATH=src` diz ao Python onde
> procurar — funciona sempre a partir da raiz, sem precisar de `cd`.

```bash
PYTHONPATH=src python -c "from data.load import load_iris_df
df = load_iris_df()
print(df.describe())
print(df.isna().sum())
print(f'Duplicados: {df.duplicated().sum()}')"
```

**Resultado esperado:**
```
       sepal_length  sepal_width  petal_length  petal_width      target
count    150.000000   150.000000    150.000000   150.000000  150.000000
mean       5.843333     3.057333      3.758000     1.199333    1.000000
std        0.828066     0.435866      1.765298     0.762238    0.819232
min        4.300000     2.000000      1.000000     0.100000    0.000000
max        7.900000     4.400000      6.900000     2.500000    2.000000
```

> 💡 **Ponto-chave**: Se `mean` está muito longe da `median`, há outliers. Se `std` é zero, a feature é constante (inútil).

✅ Estatísticas inspecionadas.

---

## 🎯 Parte 4: Detecção de Outliers

### Passo 7: Método IQR

> 📐 **O que é IQR (Interquartile Range / Amplitude Interquartil)?**
> É um jeito simples de achar **valores extremos (outliers)** olhando para o "miolo" dos dados.
>
> Como funciona, em 4 passos:
> 1. Ordena os valores do menor para o maior.
> 2. Acha dois cortes: **Q1** (25% menores) e **Q3** (25% maiores).
> 3. Calcula **IQR = Q3 − Q1** (a "largura" dos 50% do meio).
> 4. Define as cercas: **abaixo de `Q1 − 1.5×IQR`** ou **acima de `Q3 + 1.5×IQR`** = outlier.
>
> **Exemplo**: idades `[21, 22, 23, 24, 25, 26, 27, 28, 90]`. O miolo fica ~23 a 27,
> então a cerca de cima é ~33. O valor **90** passa da cerca → é outlier.
>
> **Por que é confiável**: ele usa **posições** (quartis), não a média. Por isso funciona mesmo
> quando já existe um valor maluco no meio — a média sozinha já estaria distorcida por ele.

**Adicionar a `src/data/validate.py`:**

> 📄 **O que estamos adicionando**: a função `detect_outliers_iqr`.
> **O que ela faz**: encontra os **valores extremos** de uma coluna pelo método IQR (olha os 50%
> do meio dos dados e marca o que está muito fora dessa faixa). Aqui a gente só **detecta e avisa**,
> não remove — no Iris esses extremos são naturais.

```python
def detect_outliers_iqr(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Detecta outliers usando IQR (Interquartile Range)."""
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = df[(df[column] < lower) | (df[column] > upper)]
    logger.info(f"Outliers em '{column}': {len(outliers)} / {len(df)}")
    return outliers
```

**Testar (a partir da raiz do projeto):**

```bash
PYTHONPATH=src python -c "from data.load import load_iris_df
from data.validate import detect_outliers_iqr
df = load_iris_df()
outliers = detect_outliers_iqr(df, 'sepal_width')
print(outliers)"
```

**Resultado esperado:**
```
INFO: Outliers em 'sepal_width': 4 / 150
     sepal_length  sepal_width  petal_length  petal_width  target
15            5.7          4.4           1.5          0.4       0
32            5.2          4.1           1.5          0.1       0
...
```

✅ Outliers detectados.

---

### Passo 8: Forçar Outlier e Validar Impacto

**Demonstração — impacto de um outlier (a partir da raiz do projeto):**

```bash
PYTHONPATH=src python -c "from data.load import load_iris_df
df = load_iris_df()
print(f'Média sepal_length: {df[\"sepal_length\"].mean():.2f}')
df.loc[0, 'sepal_length'] = 999.0
print(f'Média sepal_length (com outlier): {df[\"sepal_length\"].mean():.2f}')"
```

**Resultado esperado:**
```
Média sepal_length: 5.84
Média sepal_length (com outlier): 12.47
```

> ⚠️ **IMPORTANTE**: Um único outlier pode dobrar a média. Por isso validamos **antes** de treinar.

✅ Impacto do outlier demonstrado.

---

## 🔧 Troubleshooting

| Erro | Causa | Solução |
|------|-------|---------|
| `ModuleNotFoundError: data` | Falta `__init__.py` ou `pythonpath` | Criar `src/__init__.py` e configurar `pytest.ini` |
| `KeyError: 'target'` | Versão do sklearn diferente | Atualizar `requirements.txt` para sklearn 1.3+ |
| `AssertionError: Tipo errado` | Pandas leu CSV com tipo inferido errado | Use `dtype=` no `read_csv` |
| Estatísticas com `NaN` | Coluna tem valores faltantes | `df.dropna()` ou imputação (próximo vídeo) |

---

**FIM DO VÍDEO 2.1** ✅
