# Aula 04 - Cheatsheet

## Conceitos-Chave

```
Batch = Processar em lote (offline)
Real-time = Responder instantaneamente (online)
API REST = Interface HTTP para comunicação
Docker = Container isolado e reprodutível
```

## Flask API

### Estrutura Básica

```python
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Carregar modelo (uma vez, no início)
model = joblib.load('model.pkl')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Receber dados
        data = request.json
        features = np.array(data['features']).reshape(1, -1)
        
        # Predizer
        prediction = model.predict(features)
        probability = model.predict_proba(features)
        
        # Retornar
        return jsonify({
            'prediction': int(prediction[0]),
            'probability': probability[0].tolist()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)
```

### Testar API

```bash
# Testar health
curl http://localhost:5001/health

# Testar predição
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

## FastAPI (Alternativa Moderna)

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

app = FastAPI()
model = joblib.load('model.pkl')

class PredictionRequest(BaseModel):
    features: list[float]

class PredictionResponse(BaseModel):
    prediction: int
    probability: list[float]

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        features = np.array(request.features).reshape(1, -1)
        prediction = model.predict(features)
        probability = model.predict_proba(features)
        
        return PredictionResponse(
            prediction=int(prediction[0]),
            probability=probability[0].tolist()
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Rodar: uvicorn app:app --reload
```

## Docker

### Dockerfile

```dockerfile
# Imagem base
FROM python:3.9-slim

# Diretório de trabalho
WORKDIR /app

# Copiar requirements
COPY requirements.txt .

# Instalar dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código
COPY . .

# Expor porta
EXPOSE 5001

# Comando de início
CMD ["python", "app.py"]
```

### .dockerignore

```
__pycache__
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.git
.gitignore
README.md
.DS_Store
*.ipynb
.ipynb_checkpoints
```

### Comandos Docker

```bash
# Build da imagem
docker build -t ml-api:v1 .

# Listar imagens
docker images

# Rodar container
docker run -p 5001:5001 ml-api:v1

# Rodar em background
docker run -d -p 5001:5001 --name ml-api ml-api:v1

# Ver containers rodando
docker ps

# Ver logs
docker logs ml-api

# Parar container
docker stop ml-api

# Remover container
docker rm ml-api

# Entrar no container
docker exec -it ml-api /bin/bash

# Limpar tudo
docker system prune -a
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "5001:5001"
    environment:
      - MODEL_PATH=/app/models/model.pkl
    volumes:
      - ./models:/app/models
    restart: unless-stopped
```

```bash
# Rodar com compose
docker-compose up

# Rodar em background
docker-compose up -d

# Parar
docker-compose down

# Ver logs
docker-compose logs -f
```

## Serialização de Modelos

### Joblib (Recomendado)

```python
import joblib

# Salvar
joblib.dump(model, 'model.pkl', compress=3)

# Carregar
model = joblib.load('model.pkl')
```

### Pickle

```python
import pickle

# Salvar
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Carregar
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
```

### ONNX (Interoperabilidade)

```python
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Converter para ONNX
initial_type = [('float_input', FloatTensorType([None, 4]))]
onnx_model = convert_sklearn(model, initial_types=initial_type)

# Salvar
with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
```

## Validação de Input

```python
from pydantic import BaseModel, validator

class PredictionInput(BaseModel):
    features: list[float]
    
    @validator('features')
    def validate_features(cls, v):
        if len(v) != 4:
            raise ValueError('Must have exactly 4 features')
        if any(x < 0 for x in v):
            raise ValueError('Features must be non-negative')
        return v
```

## Logging

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    logger.info(f"Received prediction request")
    try:
        # ...
        logger.info(f"Prediction: {prediction}")
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 400
```

## Batch Processing

```python
import pandas as pd

def batch_predict(input_file, output_file):
    """Processar arquivo em lote."""
    # Carregar dados
    df = pd.read_csv(input_file)
    
    # Predizer
    predictions = model.predict(df[feature_cols])
    
    # Adicionar predições
    df['prediction'] = predictions
    
    # Salvar
    df.to_csv(output_file, index=False)
    
    print(f"Processed {len(df)} rows")

# Uso
batch_predict('input.csv', 'output.csv')
```

## Monitoramento

### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, generate_latest

# Métricas
prediction_counter = Counter('predictions_total', 'Total predictions')
prediction_duration = Histogram('prediction_duration_seconds', 'Prediction duration')

@app.route('/predict', methods=['POST'])
def predict():
    with prediction_duration.time():
        # ... fazer predição
        prediction_counter.inc()
        return jsonify(result)

@app.route('/metrics')
def metrics():
    return generate_latest()
```

## Boas Práticas

### ✅ Fazer
- Validar input antes de predizer
- Usar logging estruturado
- Implementar health check
- Versionar API (/v1/predict)
- Documentar API (Swagger/OpenAPI)
- Usar HTTPS em produção
- Limitar rate (rate limiting)
- Monitorar latência e erros

### ❌ Evitar
- Carregar modelo a cada requisição
- Expor stack traces em produção
- Rodar com debug=True em produção
- Hardcoded credentials
- Não validar input
- Não ter timeout
- Não ter health check

## Troubleshooting

### Erro: "Model not found"
```python
# Verificar path
import os
print(os.path.abspath('model.pkl'))
```

### Erro: "Port already in use"
```bash
# Encontrar processo
lsof -i :5001

# Matar processo
kill -9 <PID>
```

### Container não inicia
```bash
# Ver logs
docker logs <container_id>

# Entrar no container
docker run -it ml-api:v1 /bin/bash
```

---

**Última atualização**: Aula 04
