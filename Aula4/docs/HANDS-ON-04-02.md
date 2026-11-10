# 🎬 Vídeo 4.2 - Construção de API REST com Flask

**Aula**: 4 - Deploy de Modelos  
**Vídeo**: 4.2  
**Temas**: Flask; Endpoints REST; Validação; Testes com curl

---

## 🚀 Sobre Este Vídeo

> **"Flask transforma um `.pkl` em endpoint HTTP em 20 linhas de código."**

### O que você vai fazer:

| Etapa | Descrição |
|-------|-----------|
| **Hello World** | API mínima funcionando |
| **/health** | Endpoint de liveness check |
| **/predict** | Endpoint de inferência |
| **Validação** | Tratar input inválido |
| **Testar** | curl + browser |

### Pré-requisitos

| Requisito | Como verificar |
|-----------|----------------|
| Vídeo 4.1 concluído | `models/iris_model.pkl` existe |
| Flask instalado | `python -c "import flask; print(flask.__version__)"` |
| Porta 5001 livre | `lsof -i :5001` (Mac/Linux) |

> ⚠️ **macOS Sonoma+**: Porta 5000 é usada pelo AirPlay. Usaremos **5001** nos exemplos.

---

## 📚 Parte 1: API Mínima

### Passo 1: Hello World

**Criar `src/api/app.py`:**

```python
"""API REST para servir modelo de classificação."""
import logging
from flask import Flask

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


@app.route("/")
def home():
    return {"message": "ML API rodando!", "version": "1.0.0"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
```

---

### Passo 2: Subir a API

**Linux/Mac:**
```bash
mkdir -p src/api
# (criar arquivo acima)
python src/api/app.py
```

**Windows (PowerShell):**
```powershell
New-Item -ItemType Directory -Path src\api -Force
python src\api\app.py
```

**Resultado esperado:**
```
 * Running on http://127.0.0.1:5001
 * Restarting with stat
 * Debugger is active!
```

> ⚠️ **IMPORTANTE**: Deixe esse terminal aberto. Abra **outro** terminal para os testes.

---

### Passo 3: Testar

**Em outro terminal:**

**Linux/Mac:**
```bash
curl http://localhost:5001/
```

**Windows (PowerShell):**
```powershell
Invoke-RestMethod http://localhost:5001/
```

**Resultado esperado:**
```json
{"message":"ML API rodando!","version":"1.0.0"}
```

✅ API básica funcionando.

---

## 💚 Parte 2: Health Check

### Passo 4: Endpoint /health

**Atualizar `src/api/app.py`** (adicionar antes do `if __name__`):

```python
@app.route("/health", methods=["GET"])
def health():
    """Liveness check — sempre retorna 200 se a API está viva."""
    return {"status": "ok"}, 200
```

---

### Passo 5: Testar /health

**Linux/Mac:**
```bash
curl http://localhost:5001/health
```

**Windows (PowerShell):**
```powershell
Invoke-RestMethod http://localhost:5001/health
```

**Resultado esperado:**
```json
{"status":"ok"}
```

> 💡 **Ponto-chave**: `/health` é padrão DevOps. Load balancer e Kubernetes usam pra saber se o container está pronto.

✅ Health check pronto.

---

## 🧠 Parte 3: Endpoint /predict

### Passo 6: Carregar Modelo no Startup

**Atualizar `src/api/app.py`** (no topo, após imports):

```python
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

# Carrega modelo UMA VEZ no startup (não a cada request!)
MODEL_PATH = "models/iris_model.pkl"
logger.info(f"Carregando modelo de {MODEL_PATH}...")
model = joblib.load(MODEL_PATH)
logger.info("✅ Modelo carregado")

CLASSES = {0: "setosa", 1: "versicolor", 2: "virginica"}
```

> ⚠️ **CRÍTICO**: Carregar modelo a cada request mata a performance. Sempre **uma vez no startup**.

> 🧠 **Entendendo o lado ML**: pense no modelo como uma **conexão de pool / cache em memória** — você inicializa uma vez no boot e reusa em todo request. O `joblib.load` desserializa o `.pkl` (pode levar de ms a segundos dependendo do tamanho do modelo); fazer isso por request seria como abrir conexão nova de banco a cada chamada. Consequência prática pro deploy: o modelo vive na **memória do processo**, então cada worker do gunicorn carrega sua própria cópia. 4 workers = 4x a RAM do modelo. Dimensione os limites do container pensando nisso.

---

### Passo 7: Endpoint /predict

**Adicionar ao `src/api/app.py`:**

```python
@app.route("/predict", methods=["POST"])
def predict():
    """Recebe features e retorna predição."""
    data = request.get_json(silent=True)

    # 1. Validar input
    if data is None or "features" not in data:
        return jsonify({"error": "JSON inválido. Esperado: {'features': [4 floats]}"}), 400

    features = data["features"]

    if not isinstance(features, list) or len(features) != 4:
        return jsonify({"error": "features deve ser lista de 4 floats"}), 400

    # 2. Predizer
    try:
        pred = model.predict([features])[0]
        proba = model.predict_proba([features])[0]
    except Exception as e:
        return jsonify({"error": f"Erro na predição: {str(e)}"}), 500

    # 3. Resposta
    return jsonify({
        "prediction": int(pred),
        "class_name": CLASSES[int(pred)],
        "confidence": float(proba.max()),
    })
```

---

### Passo 8: Reiniciar API e Testar

Flask com `debug=True` recarrega automaticamente. Se não, mate (`Ctrl+C`) e reinicie.

**Testar predição:**

**Linux/Mac:**
```bash
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

**Windows (PowerShell):**
```powershell
$body = '{"features": [5.1, 3.5, 1.4, 0.2]}'
Invoke-RestMethod -Uri http://localhost:5001/predict -Method POST -Body $body -ContentType "application/json"
```

**Resultado esperado:**
```json
{
  "prediction": 0,
  "class_name": "setosa",
  "confidence": 1.0
}
```

✅ Predição via API funcionando.

---

### Passo 9: Testar Outras Classes

**Linux/Mac:**
```bash
# Versicolor (classe 1)
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [6.7, 3.1, 4.4, 1.4]}'

# Virginica (classe 2)
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [6.3, 3.3, 6.0, 2.5]}'
```

**Windows (PowerShell):**
```powershell
$v = '{"features": [6.7, 3.1, 4.4, 1.4]}'
Invoke-RestMethod -Uri http://localhost:5001/predict -Method POST -Body $v -ContentType "application/json"

$g = '{"features": [6.3, 3.3, 6.0, 2.5]}'
Invoke-RestMethod -Uri http://localhost:5001/predict -Method POST -Body $g -ContentType "application/json"
```

**Resultado esperado:**
```json
{"prediction":1,"class_name":"versicolor","confidence":0.97}
{"prediction":2,"class_name":"virginica","confidence":1.0}
```

✅ As 3 classes estão sendo identificadas.

---

## 🛡️ Parte 4: Validação de Erros

### Passo 10: Testar Inputs Inválidos

**Linux/Mac:**
```bash
# Sem 'features'
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{"x": [1,2,3,4]}' -w "\nStatus: %{http_code}\n"

# Lista com tamanho errado
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [1, 2]}' -w "\nStatus: %{http_code}\n"

# JSON malformado
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d 'isso não é json' -w "\nStatus: %{http_code}\n"
```

**Windows (PowerShell):**
```powershell
try {
    Invoke-RestMethod -Uri http://localhost:5001/predict -Method POST -Body '{"x":[1,2,3,4]}' -ContentType "application/json"
} catch { $_.Exception.Response.StatusCode.Value__ }
```

**Resultado esperado:**
```json
{"error":"JSON inválido. Esperado: {'features': [4 floats]}"}
Status: 400

{"error":"features deve ser lista de 4 floats"}
Status: 400

{"error":"JSON inválido. ..."}
Status: 400
```

> 💡 **Ponto-chave**: API bem feita SEMPRE valida input e retorna **HTTP 400** com mensagem clara. Nunca 500 por erro do cliente.

> 🧠 **Entendendo o lado ML**: aqui a validação é só estrutural (é lista? tem 4 elementos?). Em produção de ML existe um segundo nível que não dá exceção: o **range dos valores**. Se o modelo treinou com pétalas entre 1–7cm e chega um valor 50, o `predict` responde normalmente com uma classe qualquer e confiança alta — resposta 200, mas **silenciosamente errada**. Isso se chama *data drift / out-of-distribution*. Não precisa resolver agora, mas é o tipo de monitoramento (distribuição do input vs. baseline de treino) que entra na fase de observabilidade do modelo.

✅ Validação robusta funcionando.

---

## 🧪 Parte 5: Testes Automatizados

### Passo 11: Criar Testes

**Criar `tests/test_api.py`:**

```python
"""Testes da API."""
import pytest
from api.app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_home(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "message" in r.get_json()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.get_json() == {"status": "ok"}


def test_predict_setosa(client):
    r = client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2]})
    assert r.status_code == 200
    data = r.get_json()
    assert data["prediction"] == 0
    assert data["class_name"] == "setosa"


def test_predict_sem_features(client):
    r = client.post("/predict", json={"x": [1, 2, 3, 4]})
    assert r.status_code == 400


def test_predict_tamanho_errado(client):
    r = client.post("/predict", json={"features": [1, 2]})
    assert r.status_code == 400
```

---

### Passo 12: Rodar Testes

**Linux/Mac e Windows:**
```bash
PYTHONPATH=src pytest tests/ -v
```

> ⚠️ **`zsh: command not found: pytest`?** O `pytest` não está instalado no ambiente ativo. Ative o venv e instale antes de rodar:
> ```bash
> source venv/bin/activate
> pip install pytest          # ou: pip install -r requirements.txt
> PYTHONPATH=src pytest tests/ -v
> ```
> Lembre que ativar o venv (`source venv/bin/activate`) não instala nada — só troca o Python ativo. Se o pacote não foi instalado naquele venv, o comando continua não encontrado.

**Resultado esperado:**
```
tests/test_api.py::test_home PASSED
tests/test_api.py::test_health PASSED
tests/test_api.py::test_predict_setosa PASSED
tests/test_api.py::test_predict_sem_features PASSED
tests/test_api.py::test_predict_tamanho_errado PASSED
======================== 5 passed in 0.84s ========================
```

✅ API testada automaticamente.

---

## 🔧 Troubleshooting

| Erro | Causa | Solução |
|------|-------|---------|
| `Address already in use` | Porta 5001 ocupada | Mudar porta ou matar processo: `lsof -i :5001` |
| `Connection refused` | API não está rodando | Iniciar `python src/api/app.py` |
| Porta 5000 em uso no Mac | AirPlay Receiver | Usar 5001 ou desligar em System Preferences |
| `ModuleNotFoundError: api` | `pythonpath` errado | `PYTHONPATH=src pytest ...` |
| `command not found: pytest` | pytest não instalado no venv | Ativar venv e `pip install pytest` (ou `-r requirements.txt`) |
| `joblib.exceptions` ao carregar | Modelo treinado em sklearn diferente | Re-treinar com a versão atual |
| Flask reload não funciona | Sem `debug=True` | Habilitar debug ou reiniciar manual |

---

**FIM DO VÍDEO 4.2** ✅
