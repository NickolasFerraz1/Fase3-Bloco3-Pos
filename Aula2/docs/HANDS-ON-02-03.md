# 🎬 Vídeo 2.3 - Feature Engineering e Versionamento

**Aula**: 2 - Ingestão e Feature Engineering  
**Vídeo**: 2.3  
**Temas**: Features derivadas; Persistência; Versionamento (introdução ao DVC)

---

## 🚀 Sobre Este Vídeo

> **"Feature boa vale mais que modelo complexo."** Hoje você cria features derivadas e versiona o pipeline completo.

### O que você vai fazer:

| Etapa | Descrição |
|-------|-----------|
| **Features derivadas** | Criar features novas a partir das existentes |
| **Pipeline completo** | Encadear validate → transform → save |
| **Persistir features** | Salvar dataset processado |
| **DVC (intro)** | Versionamento de dados |

### Pré-requisitos

| Requisito | Como verificar |
|-----------|----------------|
| Aula 2.2 concluída | `src/features/transform.py` existe |
| Pytest verde | `pytest tests/` |
| Git instalado | `git --version` |

---

## 📚 Parte 1: O que é Feature Derivada?

### Passo 0: O que é "Feature Engineering"?

**Feature** é cada coluna que o modelo usa para aprender (ex.: comprimento da pétala).

**Feature Engineering** é o trabalho de **preparar e criar boas colunas** antes de treinar —
em vez de jogar o dado cru no modelo. É como **preparar bem os ingredientes** antes de cozinhar.

Ela tem 3 formas, e a Aula 02 passa pelas três:

| Forma | O que é | Onde na aula |
|-------|---------|--------------|
| **Criar** features novas | combinar colunas existentes (ex.: área = comprimento × largura) | **este vídeo (2.3)** |
| **Transformar** features | colocar na mesma escala / virar número | vídeo 2.2 |
| **Selecionar** features | manter só as que importam | vídeo 2.2 |

> 💡 **Ponto-chave**: Uma feature bem pensada costuma valer mais que um modelo complexo.

---

### Passo 1: Conceito de Feature Derivada

**Feature derivada** = nova coluna calculada a partir das existentes.

**Exemplos clássicos:**

| Dataset | Features originais | Features derivadas |
|---------|--------------------|--------------------|
| E-commerce | preço, quantidade | `valor_total = preço × quantidade` |
| RH | data_nasc, data_admissão | `idade`, `tempo_de_empresa` |
| Iris | petal_length, petal_width | `petal_area = length × width` |
| Saúde | peso, altura | `imc = peso / altura²` |

> 💡 **Ponto-chave**: Features bem desenhadas frequentemente vencem modelos sofisticados.

---

## 🛠️ Parte 2: Construir Features

### Passo 2: Criar Função de Features Derivadas

**Criar `src/features/build.py`:**

> 📄 **O que é este arquivo**: a fábrica de **features derivadas**.
> **O que ele faz**: cria colunas novas combinando as existentes — áreas (`comprimento × largura`)
> e razões entre medidas. É a feature engineering na prática: damos ao modelo informação **mais rica**
> do que as colunas cruas.

```python
"""Geração de features derivadas."""
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona features derivadas ao Iris."""
    df = df.copy()

    # Áreas (length × width)
    df["sepal_area"] = df["sepal_length"] * df["sepal_width"]
    df["petal_area"] = df["petal_length"] * df["petal_width"]

    # Razões
    df["petal_sepal_length_ratio"] = df["petal_length"] / df["sepal_length"]
    df["petal_sepal_width_ratio"] = df["petal_width"] / df["sepal_width"]

    logger.info(f"Features derivadas adicionadas. Shape: {df.shape}")
    return df
```

---

### Passo 3: Testar Geração

**Criar `tests/test_build.py`:**

> 📄 **O que é este arquivo**: os testes das features derivadas.
> **O que ele faz**: confere que as 4 colunas novas foram criadas, que o **cálculo está correto**
> (ex.: a área bate com comprimento × largura) e que a função **não altera a tabela original**
> (boa prática: cada passo entrega um resultado novo sem efeito colateral).

```python
from data.load import load_iris_df
from features.build import add_derived_features


def test_adiciona_4_features_derivadas():
    df = load_iris_df()
    df_new = add_derived_features(df)

    novas = {"sepal_area", "petal_area", "petal_sepal_length_ratio", "petal_sepal_width_ratio"}
    assert novas.issubset(df_new.columns)
    assert df_new.shape == (150, 9)  # 5 originais + 4 derivadas


def test_petal_area_calculo_correto():
    df = load_iris_df()
    df_new = add_derived_features(df)
    esperado = df.loc[0, "petal_length"] * df.loc[0, "petal_width"]
    assert df_new.loc[0, "petal_area"] == esperado
```

**Rodar:**

**Linux/Mac e Windows:**
```bash
pytest tests/test_build.py -v
```

**Resultado esperado:**
```
tests/test_build.py::test_adiciona_4_features_derivadas PASSED
tests/test_build.py::test_petal_area_calculo_correto PASSED
=================== 2 passed in 0.38s ===================
```

✅ Features derivadas geradas e testadas.

---

## 🔗 Parte 3: Pipeline Completo

### Passo 4: Orquestrador

**Criar `src/pipeline.py`:**

> 📄 **O que é este arquivo**: o **maestro** do projeto.
> **O que ele faz**: chama todos os passos na ordem certa — carrega → valida → detecta outliers →
> cria features derivadas → salva o CSV final. É o arquivo que você roda com um comando só para
> executar a esteira inteira de ponta a ponta.

```python
"""Pipeline de feature engineering completo."""
import logging
from pathlib import Path

from data.load import load_iris_df
from data.validate import validate_schema, detect_outliers_iqr
from features.build import add_derived_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("pipeline")


def run():
    logger.info("🚀 Pipeline de features iniciado")

    # 1. Carregar
    df = load_iris_df()

    # 2. Validar
    validate_schema(df)

    # 3. Detectar outliers (não removemos no Iris, só logamos)
    for col in ["sepal_length", "sepal_width", "petal_length", "petal_width"]:
        detect_outliers_iqr(df, col)

    # 4. Features derivadas
    df = add_derived_features(df)

    # 5. Persistir
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    out = "data/processed/iris_features.csv"
    df.to_csv(out, index=False)
    logger.info(f"💾 Dataset salvo em {out}")

    logger.info(f"🎉 Pipeline concluído. Shape final: {df.shape}")
    return df


if __name__ == "__main__":
    run()
```

---

### Passo 5: Executar

O pipeline executa **toda a esteira de uma vez**, na ordem:

1. **Carrega** o dado bruto (Iris).
2. **Valida** o schema (colunas e tipos certos).
3. **Detecta outliers** (só avisa, não remove — no Iris os extremos são naturais).
4. **Cria as features derivadas** (de 5 colunas vamos para 9).
5. **Salva** o resultado em CSV, pronto para a próxima aula treinar o modelo.

No log, a forma sai de `5 colunas` e termina em `Shape final: (150, 9)` — esse `9`
(5 originais + 4 derivadas) confirma que as features derivadas foram criadas.

> ⚠️ **IMPORTANTE**: Rode **a partir da raiz do projeto** (não de dentro de `src/`).
> Assim o CSV é salvo em `data/processed/` na raiz — exatamente onde o `.gitignore`
> e o DVC esperam. O Python adiciona `src/` ao path automaticamente ao executar o script.

**Linux/Mac:**
```bash
python src/pipeline.py
```

**Windows (PowerShell):**
```powershell
python src\pipeline.py
```

**Resultado esperado:**
```
... | pipeline | INFO | 🚀 Pipeline de features iniciado
... | data.validate | INFO | ✅ Schema válido: 150 linhas x 5 colunas
... | data.validate | INFO | Outliers em 'sepal_width': 4 / 150
... | features.build | INFO | Features derivadas adicionadas. Shape: (150, 9)
... | pipeline | INFO | 💾 Dataset salvo em data/processed/iris_features.csv
... | pipeline | INFO | 🎉 Pipeline concluído. Shape final: (150, 9)
```

✅ Pipeline ponta a ponta.

---

### Passo 6: Verificar Saída

**Linux/Mac:**
```bash
head -3 data/processed/iris_features.csv
wc -l data/processed/iris_features.csv
```

**Windows (PowerShell):**
```powershell
Get-Content data\processed\iris_features.csv -TotalCount 3
(Get-Content data\processed\iris_features.csv).Count
```

**Resultado esperado:**
```
sepal_length,sepal_width,petal_length,petal_width,target,sepal_area,petal_area,...
5.1,3.5,1.4,0.2,0,17.85,0.28,...
151
```

✅ Arquivo gerado com 150 linhas + header.

---

## 📦 Parte 4: Versionamento de Dados (DVC — Intro)

### Passo 7: O Problema do Dado no Git

```
data/processed/iris_features.csv      → 10 KB  (cabe no Git)
data/processed/big_dataset.csv        → 5 GB   (❌ NÃO cabe no Git)
```

> ⚠️ **IMPORTANTE**: Git foi feito para código. Dados grandes precisam de **DVC**.

---

### Passo 8: Instalar e Inicializar DVC

**Linux/Mac:**
```bash
pip install dvc
git init
dvc init
git add .dvc .dvcignore
git commit -m "feat: inicializa DVC"
```

**Windows (PowerShell):**
```powershell
pip install dvc
git init
dvc init
git add .dvc .dvcignore
git commit -m "feat: inicializa DVC"
```

**Resultado esperado:**
```
Initialized DVC repository.
You can now commit the changes to git.
```

✅ DVC inicializado.

---

### Passo 9: Versionar Dataset

**Linux/Mac e Windows:**
```bash
dvc add data/processed/iris_features.csv
git add data/processed/iris_features.csv.dvc data/processed/.gitignore
git commit -m "feat: versiona iris_features v1"
```

**Resultado esperado:**
```
Adding...
100% Add|████████████████████|1/1 [00:00, 12.34file/s]

To track the changes with git, run:
    git add data/processed/iris_features.csv.dvc data/processed/.gitignore
```

✅ Dataset versionado.

---

### Passo 10: Ver o que o DVC Criou

**Linux/Mac:**
```bash
cat data/processed/iris_features.csv.dvc
```

**Windows (PowerShell):**
```powershell
Get-Content data\processed\iris_features.csv.dvc
```

**Resultado esperado:**
```yaml
outs:
- md5: 3a4b5c6d7e8f...
  size: 8421
  hash: md5
  path: iris_features.csv
```

> 💡 **Ponto-chave**: O Git agora versiona só o `.dvc` (hash do arquivo). O dado real fica fora do Git. Na Aula 06 mostramos como configurar storage remoto (S3, GCS).

✅ Pipeline + dados versionados.

---

## 🔧 Troubleshooting

| Erro | Causa | Solução |
|------|-------|---------|
| `ModuleNotFoundError: data` | `pythonpath` errado | Rodar de `src/` ou ajustar `pytest.ini` |
| `dvc: command not found` | Não instalado | `pip install dvc` |
| `ERROR: failed to initiate DVC` | Falta `git init` antes | Rodar `git init` primeiro |
| Pipeline falha na divisão | NaN em features | `df.fillna()` ou imputar antes |
| Features derivadas com `inf` | Divisão por zero | Usar `df.replace([np.inf, -np.inf], np.nan)` |

---

**FIM DO VÍDEO 2.3** ✅

---

## 🏆 Recapitulação da Aula 02

| Vídeo | O que aprendeu |
|-------|---------------|
| **2.1** | Validação de schema, estatísticas, outliers |
| **2.2** | Normalização, encoding, Pipeline sklearn |
| **2.3** | Features derivadas, persistência, DVC |

**Próxima aula:** Treinamento de modelos, métricas e MLflow 🎯
