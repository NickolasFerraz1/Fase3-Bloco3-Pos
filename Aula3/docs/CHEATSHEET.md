# Aula 03 - Cheatsheet

## Conceitos-Chave

```
Overfitting = Decorar (não generalizar)
Underfitting = Modelo muito simples
```

## Métricas de Classificação

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report
)

# Calcular métricas
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred)

# Relatório completo
print(classification_report(y_test, y_pred))
```

### Quando Usar Cada Métrica

| Métrica | Quando Usar | Exemplo |
|---------|-------------|---------|
| **Accuracy** | Classes balanceadas | Classificar flores |
| **Precision** | Custo de falso positivo alto | Spam (não bloquear email legítimo) |
| **Recall** | Custo de falso negativo alto | Fraude (não deixar passar transação fraudulenta) |
| **F1-Score** | Balancear precision e recall | Detecção de fraude |
| **AUC-ROC** | Ranking de probabilidades | Credit scoring |

## MLflow Tracking

```python
import mlflow
import mlflow.sklearn

# Iniciar run
with mlflow.start_run():
    # Log parâmetros
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    mlflow.log_param("random_state", 42)
    
    # Treinar modelo
    model = RandomForestClassifier(n_estimators=100, max_depth=10)
    model.fit(X_train, y_train)
    
    # Log métricas
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    
    # Log modelo
    mlflow.sklearn.log_model(model, "model")
    
    # Log artefatos (gráficos, etc)
    mlflow.log_artifact("confusion_matrix.png")
```

### MLflow UI

```bash
# Iniciar UI (porta 5001 — no macOS a 5000 é usada pelo AirPlay)
mlflow ui --port 5001

# Acessar
http://localhost:5001
```

### Carregar Modelo do MLflow

```python
# Por run_id
model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

# Por nome registrado
model = mlflow.pyfunc.load_model("models:/my-model/production")
```

## Validação Cruzada

### K-Fold

```python
from sklearn.model_selection import cross_val_score

# 5-fold cross validation
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

print(f"Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

### Stratified K-Fold

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
```

### Cross Validate (múltiplas métricas)

```python
from sklearn.model_selection import cross_validate

scoring = ['accuracy', 'precision', 'recall', 'f1']
scores = cross_validate(model, X, y, cv=5, scoring=scoring)

print(f"Accuracy: {scores['test_accuracy'].mean():.3f}")
print(f"Precision: {scores['test_precision'].mean():.3f}")
print(f"Recall: {scores['test_recall'].mean():.3f}")
print(f"F1: {scores['test_f1'].mean():.3f}")
```

## Hyperparameter Tuning

### Grid Search

```python
from sklearn.model_selection import GridSearchCV

# Definir grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10]
}

# Grid Search
grid = GridSearchCV(
    estimator=RandomForestClassifier(),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)

grid.fit(X_train, y_train)

# Melhores parâmetros
print(f"Best params: {grid.best_params_}")
print(f"Best score: {grid.best_score_:.3f}")

# Melhor modelo
best_model = grid.best_estimator_
```

### Random Search

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Distribuições
param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': randint(5, 20),
    'min_samples_split': randint(2, 11),
    'min_samples_leaf': randint(1, 5)
}

# Random Search
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(),
    param_distributions=param_dist,
    n_iter=50,  # Número de combinações
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, y_train)
print(f"Best params: {random_search.best_params_}")
```

### Optuna (Avançado)

```python
import optuna

def objective(trial):
    # Sugerir hiperparâmetros
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    max_depth = trial.suggest_int('max_depth', 5, 20)
    
    # Treinar modelo
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth
    )
    
    # Cross validation
    score = cross_val_score(model, X_train, y_train, cv=5).mean()
    return score

# Otimizar
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print(f"Best params: {study.best_params}")
print(f"Best score: {study.best_value:.3f}")
```

## Learning Curves

```python
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

# Calcular learning curves
train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, cv=5, n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10)
)

# Plotar
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores.mean(axis=1), label='Train')
plt.plot(train_sizes, val_scores.mean(axis=1), label='Validation')
plt.xlabel('Training Size')
plt.ylabel('Score')
plt.legend()
plt.title('Learning Curves')
plt.show()
```

## Métricas de Regressão

```python
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, 
    r2_score, mean_absolute_percentage_error
)

# Calcular métricas
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

print(f"MSE: {mse:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"MAE: {mae:.3f}")
print(f"R²: {r2:.3f}")
print(f"MAPE: {mape:.3f}")
```

## Salvar e Carregar Modelos

```python
import joblib

# Salvar
joblib.dump(model, 'model.pkl')

# Carregar
model = joblib.load('model.pkl')

# Predizer
predictions = model.predict(X_new)
```

## Boas Práticas

### ✅ Fazer
- Sempre usar validação cruzada
- Testar múltiplas métricas
- Salvar experimentos no MLflow
- Fixar random_state para reprodutibilidade
- Usar stratified k-fold para classes desbalanceadas
- Validar no conjunto de teste apenas no final

### ❌ Evitar
- Tunar hiperparâmetros no conjunto de teste
- Usar apenas accuracy em classes desbalanceadas
- Não versionar experimentos
- Overfitting no conjunto de validação
- Grid Search com muitos parâmetros (use Random Search)

## Troubleshooting

### Overfitting
```python
# Sintomas: train score >> test score
# Soluções:
- Mais dados
- Regularização (L1, L2)
- Reduzir complexidade do modelo
- Early stopping
- Dropout (redes neurais)
```

### Underfitting
```python
# Sintomas: train score e test score baixos
# Soluções:
- Modelo mais complexo
- Mais features
- Menos regularização
- Treinar por mais tempo
```

---

**Última atualização**: Aula 03
