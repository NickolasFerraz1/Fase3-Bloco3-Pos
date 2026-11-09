"""Treina um classificador simples no dataset Iris."""
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from pathlib import Path

def main():
    # 1. Carregar dados
    print("📥 Carregando dataset Iris...")
    X, y = load_iris(return_X_y=True)
    
    # 2. Dividir treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"   Treino: {len(X_train)} | Teste: {len(X_test)}")
    
    # 3. Treinar modelo
    print("🎯 Treinando modelo...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 4. Avaliar
    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"✅ Accuracy: {accuracy:.3f}")
    
    # 5. Salvar
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, "models/iris_model.pkl")
    print("💾 Modelo salvo em models/iris_model.pkl")

if __name__ == "__main__":
    main()