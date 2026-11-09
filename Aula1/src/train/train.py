"""Treina modelo de classificação."""
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

def train_model(X, y, n_estimators=100, random_state=42):
    """Treina e retorna modelo + dados de teste."""
    logger.info(f"Treinando RandomForest com {n_estimators} árvores...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    
    logger.info(f"✅ Modelo treinado em {len(X_train)} amostras")
    return model, X_test, y_test