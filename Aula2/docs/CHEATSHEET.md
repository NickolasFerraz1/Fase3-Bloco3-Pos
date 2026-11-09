# Aula 02 - Cheatsheet

## Conceitos-Chave

```
Garbage In, Garbage Out
Qualidade dos dados > Complexidade do algoritmo
```

## Validação de Dados

```python
import pandas as pd

# Verificar tipos
df.dtypes

# Valores faltantes
df.isnull().sum()

# Estatísticas descritivas
df.describe()

# Valores únicos
df['coluna'].nunique()

# Duplicatas
df.duplicated().sum()
```

## Tratamento de Missing Values

```python
# Remover linhas com missing
df.dropna()

# Imputar com média
df['coluna'].fillna(df['coluna'].mean())

# Imputar com mediana
df['coluna'].fillna(df['coluna'].median())

# Imputar com moda
df['coluna'].fillna(df['coluna'].mode()[0])

# Forward fill
df.fillna(method='ffill')

# Criar indicador de missing
df['coluna_missing'] = df['coluna'].isnull().astype(int)
```

## Detecção de Outliers

```python
import numpy as np

# IQR method
Q1 = df['coluna'].quantile(0.25)
Q3 = df['coluna'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df[(df['coluna'] < lower) | (df['coluna'] > upper)]

# Z-score method
from scipy import stats
z_scores = np.abs(stats.zscore(df['coluna']))
outliers = df[z_scores > 3]
```

## Transformações

### Normalização (Min-Max Scaling)
```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df['coluna_norm'] = scaler.fit_transform(df[['coluna']])
```

### Padronização (Standardization)
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df['coluna_std'] = scaler.fit_transform(df[['coluna']])
```

### Log Transform
```python
df['coluna_log'] = np.log1p(df['coluna'])  # log(1 + x)
```

### Binning
```python
# Bins iguais
df['coluna_bin'] = pd.cut(df['coluna'], bins=5)

# Bins customizados
df['coluna_bin'] = pd.cut(df['coluna'], 
                           bins=[0, 18, 30, 50, 100],
                           labels=['jovem', 'adulto', 'meia_idade', 'senior'])

# Quantile-based
df['coluna_qbin'] = pd.qcut(df['coluna'], q=4)
```

## Encoding de Categóricas

### One-Hot Encoding
```python
# Pandas
df_encoded = pd.get_dummies(df, columns=['categoria'])

# Scikit-learn
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False)
encoded = encoder.fit_transform(df[['categoria']])
```

### Label Encoding
```python
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
df['categoria_encoded'] = encoder.fit_transform(df['categoria'])
```

### Target Encoding
```python
# Média do target por categoria
target_means = df.groupby('categoria')['target'].mean()
df['categoria_encoded'] = df['categoria'].map(target_means)
```

## Feature Engineering

### Features Derivadas
```python
# Idade a partir de data de nascimento
df['idade'] = (pd.Timestamp.now() - df['data_nascimento']).dt.days / 365

# Razão entre features
df['ratio'] = df['numerador'] / (df['denominador'] + 1)

# Interação
df['feature_interaction'] = df['feature1'] * df['feature2']

# Agregações
df['media_por_grupo'] = df.groupby('grupo')['valor'].transform('mean')
```

### Features Temporais
```python
df['ano'] = df['data'].dt.year
df['mes'] = df['data'].dt.month
df['dia_semana'] = df['data'].dt.dayofweek
df['dia_mes'] = df['data'].dt.day
df['trimestre'] = df['data'].dt.quarter
df['fim_semana'] = df['dia_semana'].isin([5, 6]).astype(int)
```

## Seleção de Features

### Correlação
```python
# Matriz de correlação
corr_matrix = df.corr()

# Features correlacionadas com target
corr_with_target = corr_matrix['target'].abs().sort_values(ascending=False)

# Remover features altamente correlacionadas entre si
high_corr = corr_matrix.abs() > 0.9
to_drop = [col for col in high_corr.columns if high_corr[col].sum() > 1]
```

### Feature Importance
```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Importâncias
importances = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
```

### Recursive Feature Elimination (RFE)
```python
from sklearn.feature_selection import RFE

model = RandomForestClassifier()
rfe = RFE(model, n_features_to_select=10)
rfe.fit(X_train, y_train)

selected_features = X_train.columns[rfe.support_]
```

## DVC - Versionamento de Dados

```bash
# Inicializar DVC
dvc init

# Adicionar dados ao DVC
dvc add data/raw/dataset.csv

# Commit
git add data/raw/dataset.csv.dvc data/raw/.gitignore
git commit -m "Add dataset"

# Configurar remote (S3, GCS, etc)
dvc remote add -d storage s3://mybucket/dvc-storage

# Push dados
dvc push

# Pull dados
dvc pull

# Checkout versão específica
git checkout <commit>
dvc checkout
```

## Pipeline de Feature Engineering

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Pipeline
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Fit e transform
X_transformed = pipeline.fit_transform(X_train)

# Salvar pipeline
import joblib
joblib.dump(pipeline, 'feature_pipeline.pkl')

# Carregar pipeline
pipeline = joblib.load('feature_pipeline.pkl')
X_test_transformed = pipeline.transform(X_test)
```

## Análise Estatística

```python
from scipy import stats

# Teste de normalidade
statistic, p_value = stats.normaltest(df['coluna'])

# Teste t
statistic, p_value = stats.ttest_ind(group1, group2)

# Correlação de Pearson
corr, p_value = stats.pearsonr(df['col1'], df['col2'])

# Teste qui-quadrado
chi2, p_value = stats.chi2_contingency(contingency_table)
```

## Pandas Profiling

```python
from pandas_profiling import ProfileReport

# Gerar relatório completo
profile = ProfileReport(df, title="Data Analysis Report")
profile.to_file("report.html")
```

## Boas Práticas

### ✅ Fazer
- Validar dados antes de usar
- Tratar valores faltantes conscientemente
- Normalizar/padronizar quando necessário
- Versionar dados com DVC
- Documentar transformações
- Usar pipelines para reprodutibilidade
- Testar features com validação cruzada

### ❌ Evitar
- Ignorar valores faltantes
- Normalizar sem necessidade
- Criar features sem validar utilidade
- Hardcoded transformações
- Vazamento de dados (data leakage)
- Transformar dados de teste com fit

## Troubleshooting

### Erro: "ValueError: could not convert string to float"
```python
# Verificar tipo
df['coluna'].dtype

# Converter
df['coluna'] = pd.to_numeric(df['coluna'], errors='coerce')
```

### Erro: "KeyError: column not found"
```python
# Verificar colunas
df.columns.tolist()

# Renomear se necessário
df.rename(columns={'old': 'new'}, inplace=True)
```

---

**Última atualização**: Aula 02
