"""API REST para servir modelo Iris."""
import logging
from pathlib import Path

import joblib
from flask import Flask, jsonify, request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Carrega modelo UMA VEZ no startup (nunca a cada request!)
MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "iris_model.pkl"
logger.info(f"Carregando modelo de {MODEL_PATH}...")
model = joblib.load(MODEL_PATH)
logger.info("✅ Modelo carregado")

CLASSES = {0: "setosa", 1: "versicolor", 2: "virginica"}


@app.route("/")
def home():
    return {"message": "ML API rodando!", "version": "1.0.0"}


@app.route("/health", methods=["GET"])
def health():
    """Liveness check — para load balancer / Kubernetes."""
    return {"status": "ok"}, 200


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
        logger.exception("Erro na predição")
        return jsonify({"error": f"Erro na predição: {str(e)}"}), 500

    # 3. Resposta
    return jsonify({
        "prediction": int(pred),
        "class_name": CLASSES[int(pred)],
        "confidence": float(proba.max()),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
