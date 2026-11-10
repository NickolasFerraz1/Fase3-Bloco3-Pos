# 🎬 Vídeo 6.2 - Reprodutibilidade e Model Registry

**Aula**: 6 - SOLID, Reprodutibilidade e Qualidade de Código    
**Vídeo**: 6.2  
**Temas**: Seeds; Determinismo; MLflow Tracking; Model Registry

---

## 🚀 Sobre Este Vídeo

> **"Modelo bom em janeiro, ruim em junho. Com qual versão do código e quais deps ele foi treinado? Se você não sabe, não dá pra auditar nem reproduzir."**

### O que você vai fazer:

| Etapa | Descrição |
|-------|-----------|
| **Seeds** | Fixar `random`, `numpy` e `random_state` para treino determinístico |
| **MLflow Tracking** | Logar params, métricas e o git sha do código |
| **log_model** | Salvar o binário junto do ambiente (deps) |
| **Model Registry** | Registrar versões e promover entre stages |

### Pré-requisitos

| Requisito | Como verificar |
|-----------|----------------|
| Vídeo 6.1 concluído | `src/ml_pipeline/` existe |
| Ambiente virtual ativo | `which python` aponta para o `venv` |
| MLflow instalado | `mlflow --version` |
| Repo git | `git status` funciona (para capturar o sha) |

---

## 🧰 Parte 0: Ambiente Virtual

### Passo 0: Ativar o venv

Antes de qualquer coisa, trabalhe **dentro de um ambiente virtual**. Se ainda não criou:

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

**Validar:** `which python` (ou `where python` no Windows) deve apontar para dentro de `venv/`.

> 🔍 **Nos bastidores**: o venv não é só organização — ele é parte da **reprodutibilidade**
> que este vídeo defende. O `mlflow.sklearn.log_model` captura as dependências do
> ambiente ativo (gera um `requirements.txt`/`conda.yaml` junto do modelo). Se você
> rodar fora do venv, com versões diferentes das do `requirements.txt`, o modelo
> registrado carrega o ambiente "errado" — e a promessa de reproduzir o treino meses
> depois se quebra. Ambiente isolado = registro fiel.

---

## 📚 Parte 1: Reprodutibilidade Começa na Seed

### Passo 1: Por Que Fixar Seed?

**O que é uma seed?** Computadores não geram números realmente aleatórios — usam um
gerador *pseudo*-aleatório que parte de um número inicial, a **seed** (semente). Mesma
seed → mesma sequência de números "aleatórios". É como embaralhar um baralho sempre
da mesma forma: o resultado *parece* aleatório, mas é repetível.

Sem seed fixa, o `train_test_split` e o `RandomForest` usam aleatoriedade diferente
a cada execução → métricas mudam → impossível dizer se uma variação veio do código
ou do acaso.

Veja `src/ml_pipeline/seeds.py`:

```python
SEED = 42

def set_global_seeds(seed: int = SEED) -> int:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    return seed
```

> 🔍 **Nos bastidores**: três fontes de aleatoriedade precisam ser domadas juntas.
> `PYTHONHASHSEED` controla a ordem de hashing de sets/dicts (afeta resultados que
> dependem de ordenação); `random` e `numpy` cobrem o resto. Mas atenção: o
> `random_state` **do estimador** é separado — por isso ele é cravado na factory em
> `model.py` (`RandomForestClassifier(..., random_state=seed)`). Seed global sem
> `random_state` no modelo **não** garante árvore idêntica.

---

### Passo 2: Provar o Determinismo

```bash
pytest tests/test_seeds.py -v
```

**Resultado esperado:**
```
test_mesma_seed_gera_numeros_iguais PASSED
test_treino_e_reprodutivel_com_mesma_seed PASSED
```

O teste roda a função `train()` duas vezes com a mesma seed e exige `accuracy` idêntica.

---

## 🛠️ Parte 2: MLflow Tracking

### Passo 3: Logar um Treino

O `scripts/train_register.py` envolve o treino num run do MLflow. O essencial:

```python
with mlflow.start_run() as run:
    mlflow.log_params({"model_name": ..., "test_size": ..., "seed": seed})
    mlflow.set_tag("git_sha", _git_sha())              # código exato
    result = train(model_name=..., test_size=..., seed=seed)
    mlflow.log_metrics({"accuracy": result["accuracy"],  # métricas
                        "f1_macro": result["f1_macro"]})
    mlflow.sklearn.log_model(result["model"], artifact_path="model", ...)
```

> 🔍 **Nos bastidores**: `mlflow.sklearn.log_model` não salva só o `.pkl`. Ele grava
> junto um `MLmodel`, um `requirements.txt` e um `conda.yaml` inferidos do ambiente.
> É isso que fecha o tripé da reprodutibilidade: **binário** (pickle) + **código**
> (a tag `git_sha`) + **dependências** (os requirements capturados). Sem os três,
> "carregar o modelo de janeiro" não recria janeiro.

---

### Passo 4: Executar

**Linux/Mac:**
```bash
PYTHONPATH=. python scripts/train_register.py --model rf
```

**Windows (PowerShell):**
```powershell
$env:PYTHONPATH = "."
python scripts/train_register.py --model rf
```

**Resultado esperado:**
```
Successfully registered model 'iris-classifier'.
Created version '1' of model 'iris-classifier'.
✅ run_id=...
   accuracy=0.9333 f1_macro=0.9333
   modelo registrado como 'iris-classifier'
```

> 🔍 **Nos bastidores**: por padrão o MLflow grava em `./mlruns` (backend de arquivos).
> Esse backend **não** suporta Model Registry com estágios completos em produção —
> para isso usa-se um backend com banco (ex.: `--backend-store-uri sqlite:///mlflow.db`).
> Para a aula, o arquivo local basta e o `register_model` já cria as versões.

---

### Passo 5: Ver no MLflow UI

```bash
mlflow ui --port 5001
```

Abra `http://127.0.0.1:5001`. Mostre na tela:
- A aba **Experiments** → run com params, métricas e a tag `git_sha`.
- A aba **Models** → `iris-classifier` com a **Version 1**.

> ⚠️ `mlflow ui` é um servidor de longa duração — rode num **terminal separado**
> (não trava o resto do hands-on). Encerre com `Ctrl+C` ao final.
>
> 💡 No **macOS** a porta padrão 5000 é ocupada pelo AirPlay Receiver — por isso
> usamos `--port 5001`. (Alternativa: desligar em Ajustes → Geral → AirDrop e Handoff.)

---

## 🔁 Parte 3: Versões e Reprodutibilidade

### Passo 6: Gerar uma Segunda Versão

Treine outro modelo — vira a **Version 2** automaticamente, sem sobrescrever a 1.

**Linux/Mac:**
```bash
PYTHONPATH=. python scripts/train_register.py --model logreg
```

**Windows (PowerShell):**
```powershell
$env:PYTHONPATH = "."
python scripts/train_register.py --model logreg
```

**Resultado esperado:** `Created version '2' of model 'iris-classifier'.`

> 🔍 **Nos bastidores**: o Registry versiona por **nome de modelo registrado**, não
> por arquivo. Cada `register_model` cria uma versão imutável com link para o run que
> a gerou. Promover para `Staging`/`Production` (via UI ou
> `MlflowClient().transition_model_version_stage`) é o que viabiliza rollback de
> modelo — o equivalente, para o binário, do `git checkout` para código.

---

### Passo 7: Provar a Reprodutibilidade

Rode o mesmo treino duas vezes e compare as métricas — devem ser **idênticas**, por
causa da seed fixa.

**Linux/Mac:**
```bash
PYTHONPATH=. python scripts/train_register.py --model rf
PYTHONPATH=. python scripts/train_register.py --model rf
```

**Windows (PowerShell):**
```powershell
$env:PYTHONPATH = "."
python scripts/train_register.py --model rf
python scripts/train_register.py --model rf
```

Ambos imprimem `accuracy=0.9333 f1_macro=0.9333`. Esse é o ponto: com a seed fixada
em `set_global_seeds(seed)`, o split e o modelo são determinísticos — o resultado se
repete em qualquer máquina.

> 🔍 **Nos bastidores**: se a seed **não** fosse fixada, cada execução sortearia um
> split treino/teste diferente e um `RandomForest` diferente, e as métricas oscilariam
> a cada run. É exatamente esse "ruído" que a seed elimina — sem ele, você não consegue
> dizer se uma variação veio de uma mudança no código ou só do acaso.

---

## 🔧 Troubleshooting

| Erro | Causa | Solução |
|------|-------|---------|
| Métricas mudam entre runs | Seed não aplicada | Chamar `set_global_seeds` antes do treino |
| `git_sha = unknown` | Fora de um repo git | Rodar dentro do repositório |
| Registry não aparece | Backend de arquivos | Usar `mlflow ui --port 5001 --backend-store-uri sqlite:///mlflow.db` |
| `mlflow ui` trava o terminal | É servidor | Rodar em terminal separado |
| Porta 5000 ocupada (macOS) | AirPlay Receiver usa a 5000 | `mlflow ui --port 5001` |
| `DeprecationWarning: artifact_path` | MLflow 3.x instalado | Usar `mlflow==2.9.2` (pin da aula) |

---

**FIM DO VÍDEO 6.2** ✅
