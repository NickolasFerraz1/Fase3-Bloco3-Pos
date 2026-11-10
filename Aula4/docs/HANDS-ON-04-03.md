# 🎬 Vídeo 4.3 - Ambientes Reprodutíveis com Docker

**Aula**: 4 - Deploy de Modelos  
**Vídeo**: 4.3  
**Temas**: Dockerfile; Build; Run; Docker Compose; Multi-stage

---

## 🚀 Sobre Este Vídeo

> **"'Funciona na minha máquina' é problema do passado. Docker = pacote auto-contido."**

### O que você vai fazer:

| Etapa | Descrição |
|-------|-----------|
| **Dockerfile** | Escrever receita do container |
| **Build** | Construir imagem |
| **Run** | Rodar API em container |
| **Compose** | Orquestrar múltiplos serviços |
| **Otimizar** | Imagem menor com multi-stage |

### Pré-requisitos

| Requisito | Como verificar |
|-----------|----------------|
| Vídeo 4.2 concluído | API Flask rodando localmente |
| Docker Desktop instalado | `docker --version` |
| Docker rodando | `docker ps` (sem erro) |
| Porta 5001 livre | `lsof -i :5001` |

---

## 📚 Parte 1: Por Que Docker?

### Passo 1: O Problema

| Sem Docker | Com Docker |
|------------|------------|
| Aluno: "tô com Python 3.9" | `FROM python:3.9-slim` em todos |
| "Esqueci de instalar X" | `requirements.txt` no build |
| Deploy em ECS/K8s diferente | Imagem é a mesma em tudo |
| "Tem que instalar nvidia driver" | Imagem já tem tudo |

> 💡 **Ponto-chave**: Container é o **artefato final** do MLOps. O que você deploya é a **imagem**, não o código.

---

## 🐳 Parte 2: Primeiro Dockerfile

### Passo 2: Criar Dockerfile

**Criar `Dockerfile` na raiz do projeto:**

```dockerfile
# 1. Base
FROM python:3.9-slim

# 2. Diretório de trabalho
WORKDIR /app

# 3. Instalar dependências PRIMEIRO (cache!)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copiar código
COPY src/ ./src/

# 5. Treinar o modelo DENTRO da imagem
#    Garante que o .pkl nasce com as MESMAS versões que vão executá-lo
RUN python src/train.py

# 6. Expor porta
EXPOSE 5001

# 7. Variáveis de ambiente
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# 8. Comando padrão
CMD ["python", "src/api/app.py"]
```

> 💡 **Boa prática**: copiar `requirements.txt` ANTES do código. Docker reaproveita layers — se só o código mudou, não reinstala deps.

> 🧠 **Por que treinar o modelo no build (e não `COPY models/`)?** Um `.pkl` carrega dentro de si referências às versões de `scikit-learn`/`numpy` usadas no treino. Se você treina num ambiente (ex.: sklearn 1.9 / numpy 2.x) e o container roda outro (sklearn 1.3 / numpy 1.24), o `joblib.load` pode falhar com `InconsistentVersionWarning` ou até `ModuleNotFoundError: No module named 'numpy._core'` — a API nem sobe. Treinar com `RUN python src/train.py` faz o artefato nascer com as versões exatas do `requirements.txt`, eliminando a divergência na raiz. Em projeto real, o treino fica num pipeline separado e o `.pkl` versionado vem de um artifact store (MLflow/S3) com as versões registradas — mas o princípio é o mesmo: **artefato e runtime precisam casar**.

> 🧠 **Atenção em produção**: o `CMD` acima sobe o **servidor de desenvolvimento do Flask** (single-thread, `debug=True` no `app.py`). Ótimo pra aula, proibido em produção. O `requirements.txt` já traz o `gunicorn` justamente pra isso — em produção o `CMD` seria algo como:
> ```dockerfile
> CMD ["gunicorn", "--bind", "0.0.0.0:5001", "--workers", "4", "api.app:app"]
> ```
> Cada `worker` é um processo que carrega sua própria cópia do modelo na RAM (ver nota do 4.2). Regra prática de dimensionamento: `workers = 2 × núcleos + 1`, mas ajuste pela RAM que o modelo consome.

---

### Passo 3: Criar .dockerignore

**Criar `.dockerignore`:**

```
__pycache__/
*.pyc
venv/
.git/
.pytest_cache/
tests/
data/
notebooks/
*.md
.env
```

> ⚠️ **IMPORTANTE**: Sem `.dockerignore`, o build manda **toda a pasta** (incluindo `venv` de 500MB) pro Docker. Pega o que não precisa.

---

### Passo 4: Build

**Linux/Mac:**
```bash
docker build -t ml-api:v1 .
```

**Windows (PowerShell):**
```powershell
docker build -t ml-api:v1 .
```

**Resultado esperado:**
```
[+] Building 14.7s (11/11) FINISHED
 => [internal] load build definition from Dockerfile
 => [1/6] FROM python:3.9-slim
 => [2/6] WORKDIR /app
 => [3/6] COPY requirements.txt .
 => [4/6] RUN pip install --no-cache-dir -r requirements.txt
 => [5/6] COPY src/ ./src/
 => [6/6] RUN python src/train.py
 => exporting to image
 => => naming to docker.io/library/ml-api:v1
```

✅ Imagem construída.

---

### Passo 5: Verificar Imagem

**Linux/Mac e Windows:**
```bash
docker images | grep ml-api
```

**Resultado esperado:**
```
ml-api    v1    abc123def456   30 seconds ago   180MB
```

> 💡 **Reparou?** Python slim deixa imagem em ~180MB. `python:3.9` cheia seria ~900MB.

✅ Imagem disponível.

---

## ▶️ Parte 3: Rodar Container

### Passo 6: Run

**Linux/Mac:**
```bash
docker run -d -p 5001:5001 --name ml-api ml-api:v1
```

**Windows (PowerShell):**
```powershell
docker run -d -p 5001:5001 --name ml-api ml-api:v1
```

**Resultado esperado:** Hash do container (12 caracteres).

---

### Passo 7: Validar

**Linux/Mac:**
```bash
docker ps
curl http://localhost:5001/health
```

**Windows (PowerShell):**
```powershell
docker ps
Invoke-RestMethod http://localhost:5001/health
```

**Resultado esperado:**
```
CONTAINER ID   IMAGE       PORTS                    NAMES
abc123def456   ml-api:v1   0.0.0.0:5001->5001/tcp   ml-api

{"status":"ok"}
```

✅ API rodando dentro de container.

---

### Passo 8: Predição via Container

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
{"prediction": 0, "class_name": "setosa", "confidence": 1.0}
```

✅ Modelo servindo via container.

---

### Passo 9: Ver Logs

**Linux/Mac e Windows:**
```bash
docker logs ml-api
```

**Resultado esperado:**
```
INFO:__main__:Carregando modelo de /app/models/iris_model.pkl...
INFO:__main__:✅ Modelo carregado
 * Running on http://0.0.0.0:5001
```

✅ Logs centralizados.

---

## 🐙 Parte 4: Docker Compose

### Passo 10: Por Que Compose?

Em produção, você não roda só **uma** coisa. Quer também:
- MLflow (tracking)
- PostgreSQL (backend MLflow)
- API

Docker Compose orquestra **múltiplos containers** com uma config YAML.

---

### Passo 11: Criar docker-compose.yml

**Criar `docker-compose.yml`:**

```yaml
services:
  api:
    build: .
    container_name: ml-api
    ports:
      - "5001:5001"
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000
    depends_on:
      - mlflow
    restart: unless-stopped

  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.8.1
    container_name: ml-mlflow
    ports:
      # Host 5002 → container 5000 (a 5000 do host é do AirPlay no macOS Sonoma+)
      - "5002:5000"
    command: mlflow server --host 0.0.0.0 --port 5000
    volumes:
      - ./mlruns:/mlruns
    restart: unless-stopped
```

> 💡 **Por que `5002:5000` e o `MLFLOW_TRACKING_URI` continua em `:5000`?** O mapeamento `host:container` só vale pra quem acessa **de fora** (seu browser → `localhost:5002`). A API fala com o MLflow **dentro da rede do compose**, pelo nome do serviço (`http://mlflow:5000`), e esse tráfego não passa pelo host — por isso continua na 5000.

---

### Passo 12: Subir Stack Completa

**Linux/Mac:**
```bash
docker stop ml-api && docker rm ml-api  # parar container anterior
docker compose up -d
docker compose ps
```

**Windows (PowerShell):**
```powershell
docker stop ml-api; docker rm ml-api
docker compose up -d
docker compose ps
```

**Resultado esperado:**
```
NAME         IMAGE           STATUS          PORTS
ml-api       fiap-aula04-api  Up 10 seconds   0.0.0.0:5001->5001/tcp
ml-mlflow    mlflow:v2.8.1   Up 12 seconds   0.0.0.0:5002->5000/tcp
```

✅ API + MLflow rodando juntos.

---

### Passo 13: Validar Stack

**Linux/Mac:**
```bash
curl http://localhost:5001/health
curl http://localhost:5002/ -I  # MLflow
```

**Windows (PowerShell):**
```powershell
Invoke-RestMethod http://localhost:5001/health
Invoke-WebRequest http://localhost:5002/ -Method Head | Select-Object StatusCode
```

**Resultado esperado:**
```
{"status":"ok"}
StatusCode: 200    # MLflow respondendo
```

**Abrir MLflow no browser**: http://localhost:5002

✅ Stack completa.

---

## 🪶 Parte 5: Multi-Stage Build

### Passo 14: Imagem Menor

**Criar `Dockerfile.slim`:**

```dockerfile
# Stage 1: Builder
FROM python:3.9-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime
FROM python:3.9-slim

WORKDIR /app

# Copia só as dependências instaladas
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copia código
COPY src/ ./src/

# Treina o modelo na própria imagem (versões de runtime idênticas)
RUN python src/train.py

EXPOSE 5001
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

CMD ["python", "src/api/app.py"]
```

**Build e comparar:**

**Linux/Mac e Windows:**
```bash
docker build -f Dockerfile.slim -t ml-api:slim .
docker images | grep ml-api
```

**Resultado esperado:**
```
ml-api    v1      abc...   180MB
ml-api    slim    def...   145MB
```

> 💡 **Ponto-chave**: Multi-stage descarta artefatos de build, deixa só o necessário pra rodar.

> 🧠 **Entendendo o lado ML**: o `.pkl` é serializado com uma versão específica do scikit-learn. Carregar com versão diferente pode dar `InconsistentVersionWarning` ou quebrar silenciosamente. Por isso o `scikit-learn==1.3.0` está **pinado** no `requirements.txt` — o container garante que a versão de runtime é idêntica à de treino. Essa é a real razão de "containerizar modelo": não é só o código, é congelar o **par (código + versões das libs de ML)** que produziu aquele `.pkl`.

✅ Imagem otimizada.

---

## 🧹 Parte 6: Limpeza

### Passo 15: Parar e Limpar

**Linux/Mac e Windows:**
```bash
docker compose down
docker rmi ml-api:v1 ml-api:slim
```

**Resultado esperado:**
```
[+] Running 3/3
 ✔ Container ml-api      Removed
 ✔ Container ml-mlflow   Removed
 ✔ Network fiap-aula04_default  Removed
Untagged: ml-api:v1
Untagged: ml-api:slim
```

✅ Ambiente limpo.

---

## 🔧 Troubleshooting

| Erro | Causa | Solução |
|------|-------|---------|
| `Cannot connect to Docker daemon` | Docker Desktop não está rodando | Abrir Docker Desktop |
| `Bind: address already in use` | Porta 5001 ocupada por outro container | `docker stop $(docker ps -q)` |
| `ports are not available: ... 0.0.0.0:5000 ... address already in use` | AirPlay (ControlCenter) ocupa a 5000 no macOS | Mapear MLflow como `5002:5000` no compose, ou desligar AirPlay Receiver em System Settings |
| `exec format error` no Mac M1/M2 | Imagem amd64 em arm64 | Usar `--platform linux/amd64` no build |
| Build muito lento | Sem `.dockerignore` | Criar `.dockerignore` (Passo 3) |
| `OOMKilled` no container | Pouca memória pro modelo | Aumentar memória do Docker Desktop |
| `ModuleNotFoundError` dentro do container | `PYTHONPATH` errado | Verificar `ENV PYTHONPATH=/app/src` |
| `No module named 'numpy._core'` ao carregar `.pkl` | Modelo treinado com numpy/sklearn diferente do runtime | Treinar no build (`RUN python src/train.py`) em vez de `COPY models/` |
| `InconsistentVersionWarning` ao carregar `.pkl` | sklearn do treino ≠ sklearn do container | Mesmo fix: treinar dentro da imagem ou alinhar versões no `requirements.txt` |

---

**FIM DO VÍDEO 4.3** ✅

---

## 🏆 Recapitulação da Aula 04

| Vídeo | O que aprendeu |
|-------|---------------|
| **4.1** | Batch vs Tempo Real: quando usar cada um |
| **4.2** | API REST com Flask, health check, validação |
| **4.3** | Dockerfile, build, run, Compose, multi-stage |

**Próxima aula:** Orquestração de pipelines com Apache Airflow 🚀
