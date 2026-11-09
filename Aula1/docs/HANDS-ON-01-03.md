# 🎬 Vídeo 1.3 - Testes e Reprodutibilidade

**Aula**: 1 - Introdução ao Pipeline de ML  
**Vídeo**: 1.3  
**Temas**: Testes automatizados; Pytest; Reprodutibilidade; Random seed

---

## 🚀 Sobre Este Vídeo

> **Pipeline sem testes = pipeline quebrado em produção.** Vamos adicionar testes automatizados.

### O que você vai fazer:

| Etapa | Descrição |
|-------|-----------|
| **Pytest** | Instalar framework de testes |
| **Testar módulos** | Teste para cada etapa |
| **Cobertura** | Medir % de código testado |
| **Reprodutibilidade** | Garantir resultados iguais |

### Pré-requisitos

| Requisito | Como verificar |
|-----------|----------------|
| Aula 1.2 concluída | `src/pipeline.py` existe |
| venv ativado | Prompt mostra `(venv)` |
| Pipeline rodando | `python src/pipeline.py` funciona |

---

## 📚 Parte 1: Por que Testar?

### Passo 1: O Problema

**Cenário sem testes:**

| Situação | Resultado |
|----------|-----------|
| Mudou versão do sklearn | Modelo quebra em produção |
| Alterou função `validate_data()` | Dados ruins passam |
| Refatorou `train_model()` | Accuracy cai 20% |
| Deploy automático | Quebra silenciosa |

**Com testes:**
- ✅ CI bloqueia merge se quebrar
- ✅ Refatora com segurança
- ✅ Documentação viva do que o código faz

> 💡 **Ponto-chave**: Em ML, você testa **código E modelo**. Não basta accuracy alta — o pipeline precisa ser testável.

---

## 🛠️ Parte 2: Setup do Pytest

### Passo 2: Instalar Pytest

> `pytest` e `pytest-cov` já estão no `requirements.txt` da Aula 01. Rode o comando abaixo apenas se ainda não os tiver instalado.

**Linux/Mac:**
```bash
pip install pytest pytest-cov
```

**Windows (PowerShell):**
```powershell
pip install pytest pytest-cov
```

**Resultado esperado:**
```
Requirement already satisfied: pytest
```
(ou `Successfully installed pytest-7.4.0 pytest-cov-4.1.0`)

✅ Pytest instalado.

---

### Passo 3: Estrutura de Testes

**Linux/Mac:**
```bash
mkdir -p tests
touch tests/__init__.py tests/test_data.py tests/test_train.py tests/test_pipeline.py
ls tests/
```

**Windows (PowerShell):**
```powershell
New-Item -ItemType Directory -Path tests -Force
New-Item -ItemType File -Path tests\__init__.py, tests\test_data.py, tests\test_train.py, tests\test_pipeline.py -Force
Get-ChildItem tests\
```

**Resultado esperado:**
```
__init__.py  test_data.py  test_pipeline.py  test_train.py
```

✅ Estrutura de testes criada.

---

### Passo 4: Configurar Pytest

**Criar `pytest.ini` na raiz:**

```ini
[pytest]
testpaths = tests
pythonpath = src
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short
```

> 💡 **Ponto-chave**: `pythonpath = src` permite importar `from data.ingest import ...`

---

## 🧪 Parte 3: Escrever Testes

### Passo 5: Teste do Módulo de Dados

**Criar `tests/test_data.py`:**

```python
"""Testes para data/ingest.py e data/validate.py."""
import pytest
import numpy as np
from data.ingest import load_data
from data.validate import validate_data

class TestIngest:
    def test_load_data_retorna_X_e_y(self):
        X, y = load_data()
        assert X is not None
        assert y is not None
    
    def test_load_data_tamanho_correto(self):
        X, y = load_data()
        assert X.shape == (150, 4)
        assert y.shape == (150,)
    
    def test_load_data_3_classes(self):
        _, y = load_data()
        assert len(np.unique(y)) == 3


class TestValidate:
    def test_dados_validos_passam(self):
        X, y = load_data()
        assert validate_data(X, y) is True
    
    def test_dataset_vazio_falha(self):
        with pytest.raises(AssertionError, match="Dataset vazio"):
            validate_data(np.array([]).reshape(0, 4), np.array([]))
    
    def test_x_y_tamanhos_diferentes_falha(self):
        with pytest.raises(AssertionError, match="tamanhos diferentes"):
            validate_data(np.ones((10, 4)), np.ones(5))
```

---

### Passo 6: Teste do Treinamento

**Criar `tests/test_train.py`:**

```python
"""Testes para train/train.py e train/evaluate.py."""
import pytest
from data.ingest import load_data
from train.train import train_model
from train.evaluate import evaluate_model

@pytest.fixture
def dados():
    """Fixture compartilhada."""
    return load_data()

class TestTrain:
    def test_modelo_treina_sem_erro(self, dados):
        X, y = dados
        model, X_test, y_test = train_model(X, y)
        assert model is not None
    
    def test_modelo_e_reproduzivel(self, dados):
        """Mesmo random_state → mesmo modelo."""
        X, y = dados
        m1, _, _ = train_model(X, y, random_state=42)
        m2, _, _ = train_model(X, y, random_state=42)
        # Mesma predição para mesmo input
        assert (m1.predict(X) == m2.predict(X)).all()


class TestEvaluate:
    def test_accuracy_acima_de_90(self, dados):
        X, y = dados
        model, X_test, y_test = train_model(X, y)
        metrics = evaluate_model(model, X_test, y_test)
        assert metrics["accuracy"] >= 0.90
    
    def test_retorna_dict_com_accuracy(self, dados):
        X, y = dados
        model, X_test, y_test = train_model(X, y)
        metrics = evaluate_model(model, X_test, y_test)
        assert "accuracy" in metrics
```

---

### Passo 7: Teste do Pipeline Completo

**Criar `tests/test_pipeline.py`:**

```python
"""Teste end-to-end do pipeline."""
from pathlib import Path
from pipeline import run_pipeline

def test_pipeline_executa_completo(tmp_path, monkeypatch):
    """Pipeline roda do início ao fim e gera modelo."""
    monkeypatch.chdir(tmp_path)
    
    metrics = run_pipeline()
    
    # Modelo foi salvo
    assert Path("models/iris_model.pkl").exists()
    
    # Accuracy aceitável
    assert metrics["accuracy"] >= 0.85
```

---

## ▶️ Parte 4: Executar Testes

### Passo 8: Rodar Todos os Testes

**Linux/Mac:**
```bash
pytest
```

**Windows (PowerShell):**
```powershell
pytest
```

**Resultado esperado:**
```
============================ test session starts =============================
collected 11 items

tests/test_data.py::TestIngest::test_load_data_retorna_X_e_y PASSED  [  9%]
tests/test_data.py::TestIngest::test_load_data_tamanho_correto PASSED [ 18%]
tests/test_data.py::TestIngest::test_load_data_3_classes PASSED      [ 27%]
tests/test_data.py::TestValidate::test_dados_validos_passam PASSED   [ 36%]
tests/test_data.py::TestValidate::test_dataset_vazio_falha PASSED    [ 45%]
tests/test_data.py::TestValidate::test_x_y_tamanhos_diferentes_falha PASSED [54%]
tests/test_train.py::TestTrain::test_modelo_treina_sem_erro PASSED   [ 63%]
tests/test_train.py::TestTrain::test_modelo_e_reproduzivel PASSED    [ 72%]
tests/test_train.py::TestEvaluate::test_accuracy_acima_de_90 PASSED  [ 81%]
tests/test_train.py::TestEvaluate::test_retorna_dict_com_accuracy PASSED [90%]
tests/test_pipeline.py::test_pipeline_executa_completo PASSED        [100%]

============================ 11 passed in 2.34s ==============================
```

✅ 11/11 testes passando!

---

### Passo 9: Medir Cobertura de Testes

**Linux/Mac:**
```bash
pytest --cov=src --cov-report=term-missing
```

**Windows (PowerShell):**
```powershell
pytest --cov=src --cov-report=term-missing
```

**Resultado esperado:**
```
--------- coverage: platform darwin, python 3.11 ----------
Name                    Stmts   Miss  Cover   Missing
-----------------------------------------------------
src/data/ingest.py          8      0   100%
src/data/validate.py       11      0   100%
src/train/train.py         11      0   100%
src/train/evaluate.py      10      0   100%
src/pipeline.py            22      1    95%   52
-----------------------------------------------------
TOTAL                      62      1    98%
```

✅ 98% de cobertura — excelente!

---

### Passo 10: Gerar Relatório HTML

**Linux/Mac:**
```bash
pytest --cov=src --cov-report=html
open htmlcov/index.html
```

**Windows (PowerShell):**
```powershell
pytest --cov=src --cov-report=html
Start-Process htmlcov\index.html
```

**Resultado esperado:** Browser abre com relatório visual de cobertura.

✅ Relatório HTML gerado.

---

## 🔁 Parte 5: Reprodutibilidade

### Passo 11: Por que `random_state=42`?

> ⚠️ **IMPORTANTE**: ML usa aleatoriedade (inicialização de pesos, split de dados, etc.). Sem fixar a seed, **mesmo código gera modelos diferentes**.

**Sem random_state (RUIM):**
```python
# Execução 1: accuracy = 0.967
# Execução 2: accuracy = 0.933  ❌ Diferente!
model = RandomForestClassifier()
```

**Com random_state (BOM):**
```python
# Execução 1: accuracy = 0.967
# Execução 2: accuracy = 0.967  ✅ Igual!
model = RandomForestClassifier(random_state=42)
```

---

### Passo 12: Checklist de Reprodutibilidade

| Item | Como garantir |
|------|---------------|
| **Random seed** | `random_state=42` em todos os componentes |
| **Versões fixas** | `requirements.txt` com `==` (não `>=`) |
| **Dados** | Versionar com DVC (Aula 06) |
| **Ambiente** | Docker (Aula 04) |
| **Código** | Git commit hash |

✅ Pipeline reprodutível.

---

## 🔧 Troubleshooting

| Erro | Causa | Solução |
|------|-------|---------|
| `ModuleNotFoundError: data` ao rodar pytest | `pythonpath` não configurado | Verificar `pytest.ini` (Passo 4) |
| `pytest: command not found` | venv não ativado | Reativar venv |
| Cobertura 0% | `--cov=src` errado | Usar caminho correto da pasta de código |
| Testes passam local, falham no CI | Versões diferentes | Fixar versões em `requirements.txt` |
| Modelo com accuracy diferente cada run | Sem `random_state` | Adicionar `random_state=42` em todos os componentes |

---

**FIM DO VÍDEO 1.3** ✅

---

## 🏆 Recapitulação da Aula 01

| Vídeo | O que aprendeu |
|-------|---------------|
| **1.1** | Setup ambiente + primeiro modelo |
| **1.2** | Pipeline modular com logging |
| **1.3** | Testes automatizados + reprodutibilidade |

**Próxima aula:** Ingestão de Dados e Feature Engineering 🚀
