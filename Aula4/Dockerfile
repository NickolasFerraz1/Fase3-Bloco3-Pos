FROM python:3.12-slim

WORKDIR /app

# Instalar dependências primeiro (cache de layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código
COPY src/ ./src/

# Treinar o modelo DENTRO da imagem (mesmas versões de runtime → sem InconsistentVersionWarning)
RUN python src/train.py

EXPOSE 5001

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# Health check do Docker
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5001/health')" || exit 1

CMD ["python", "src/api/app.py"]
