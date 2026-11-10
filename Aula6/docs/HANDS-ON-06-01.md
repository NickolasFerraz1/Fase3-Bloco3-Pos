# 🎬 Vídeo 6.1 - Princípios SOLID em Projetos de ML

**Aula**: 6 - SOLID, Reprodutibilidade e Qualidade de Código    
**Vídeo**: 6.1  
**Temas**: SOLID (SRP e OCP); Complexidade ciclomática; Refatoração

---

## 🚀 Sobre Este Vídeo

> **"Notebook vira `train.py` de 600 linhas que ninguém ousa tocar. Engenharia de software resolve isso — inclusive em ML."**

Vamos focar em **dois** princípios SOLID, os mais visuais e diretos:

| Princípio | Em uma frase |
|-----------|--------------|
| **SRP** (Single Responsibility) | Cada função faz **uma** coisa |
| **OCP** (Open/Closed) | Adicionar algo novo = **acrescentar**, não reescrever |

### O que você vai fazer:

| Etapa | Descrição |
|-------|-----------|
| **Medir** | Complexidade da "função monolítica" com `radon` |
| **Refatorar** | Quebrar em funções pequenas (`data`, `model`, `pipeline`) |
| **Comparar** | Medir de novo e ver a complexidade despencar |

### Pré-requisitos

| Requisito | Como verificar |
|-----------|----------------|
| Python 3.11 | `python3.11 --version` |

---

## 🧰 Parte 0: Ambiente Virtual

### Passo 0: Ativar o venv

Trabalhe sempre **dentro de um ambiente virtual**. Crie uma vez:

**Linux/Mac:**
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

A cada novo terminal, basta reativar o venv:

**Linux/Mac:**
```bash
source venv/bin/activate
```

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

> ⚠️ Sem o venv ativo você verá `command not found: radon` (ou `pylint`). O prompt
> deve mostrar `(venv)` no começo. Valide com `which python` — tem que apontar para `venv/`.

---

## 📚 Parte 1: O Problema

### Passo 1: A "Função Monolítica"

Abra `src/before/train_monolitico.py`. É uma única função `run_everything()` que
carrega dados, valida, limpa, remove outliers num loop manual, faz split, escolhe o
modelo por `if/elif`, treina, avalia, imprime relatório e salva em disco.

Funciona. E é esse o problema: funciona, então ninguém refatora — até precisar mudar
uma regra e quebrar três coisas sem querer.

| Sintoma | O que está errado |
|---------|-------------------|
| Faz 9 coisas diferentes | Viola **SRP** |
| Adicionar um modelo = editar a função | Viola **OCP** |
| Impossível testar o "limpar" sem rodar o "treinar" | Tudo acoplado |

---

### Passo 2: Medir a Complexidade Ciclomática

```bash
radon cc src/before/train_monolitico.py -s
```

**Resultado esperado:**
```
src/before/train_monolitico.py
    F 22:0 run_everything - D (23)
```

> 🔍 **Nos bastidores**: complexidade ciclomática conta os caminhos independentes de
> execução — na prática, `1 + (número de if/elif/for/while/and/or)`. O `radon` vai de
> **A** (≤5, simples) a **F** (>40, intratável). Um **D (23)** numa única função são 23
> caminhos para cobrir em teste e 23 lugares para um bug se esconder. A meta é puxar
> tudo para a faixa **A**.

---

## 🛠️ Parte 2: Refatorar (SRP + OCP)

A ideia é simples: **uma tarefa por função**, em arquivos separados por assunto.

```
src/ml_pipeline/
├── data.py       # load_data() e clean_data()   -> SRP (uma função, uma tarefa)
├── model.py      # build_model() + dict MODELS   -> OCP (novo modelo = +1 linha)
└── pipeline.py   # train() chama tudo na ordem
```

### Passo 3: Separar os Dados (SRP)

`src/ml_pipeline/data.py` tem duas funções, cada uma com uma responsabilidade:
`load_data` (traz o dataset) e `clean_data` (limpa). Nenhuma treina ou imprime.

```python
def load_data(csv_path="data/iris.csv"):
    ...

def clean_data(df):
    ...
```

---

### Passo 4: Isolar a Escolha do Modelo (OCP)

`src/ml_pipeline/model.py` guarda os modelos num **dicionário**. Para adicionar um
modelo novo, você acrescenta uma linha — não mexe em mais nada.

```python
MODELS = {
    "rf": lambda seed: RandomForestClassifier(n_estimators=100, random_state=seed),
    "logreg": lambda seed: LogisticRegression(max_iter=200, random_state=seed),
}

def build_model(name, seed=42):
    if name not in MODELS:
        raise ValueError(...)
    return MODELS[name](seed)
```

> 🔍 **Nos bastidores**: isto é OCP na prática. O `pipeline.py` chama `build_model("rf")`
> sem saber quais modelos existem. Adicionar um terceiro, um quarto modelo nunca obriga
> a tocar no código de treino — só na "lista". O `random_state=seed` cravado aqui já
> conecta com a reprodutibilidade do Vídeo 6.2.

---

### Passo 5: Juntar na Ordem

`src/ml_pipeline/pipeline.py` tem uma função `train()` que chama as peças em sequência:
carrega → limpa → divide → treina → avalia. Curta e linear, fácil de ler.

```python
def train(model_name="rf", test_size=0.2, seed=42):
    df = clean_data(load_data())
    ...
    return {"model": model, "accuracy": ..., "f1_macro": ..., "n_rows": ...}
```

---

## 📊 Parte 3: Comparar

### Passo 6: Medir de Novo

```bash
radon cc src/ml_pipeline/ -s -a
```

**Resultado esperado (resumido):**
```
5 blocks (classes, functions, methods) analyzed.
Average complexity: A (1.6)
```

| Métrica | Antes (monolítico) | Depois (ml_pipeline) |
|---------|----------------------|----------------------|
| Pior bloco | **D (23)** | **A** |
| Média | — | **A (1.6)** |
| Testável em partes | Não | Sim |
| Adicionar modelo | Editar a função | +1 linha no dicionário |

> 🔍 **Nos bastidores**: a complexidade não some — ela se **distribui**. Cada caminho
> agora vive numa função pequena e nomeada. Bug na limpeza? Está em `clean_data`, com
> teste próprio. É isso que reduz o custo de manutenção quando a equipe cresce.

---

### Passo 7: Provar com Testes

A refatoração só "vale" se cada peça puder ser testada isolada. Veja `tests/test_pipeline.py`.

```bash
pytest tests/test_pipeline.py -v
```

**Resultado esperado:** todos os testes passando (load, clean, build_model, train).

---

## 🔧 Troubleshooting

| Erro | Causa | Solução |
|------|-------|---------|
| `radon: command not found` | Não instalado | `pip install radon` |
| `ModuleNotFoundError: src` | Rodando fora da raiz | Rode da raiz com `PYTHONPATH=.` |
| `ImportError` circular | Import entre módulos do pipeline | Dependências num sentido só (pipeline → data/model) |

---

**FIM DO VÍDEO 6.1** ✅
