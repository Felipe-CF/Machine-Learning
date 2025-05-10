# Análise dos desempenhos - Algortimos, Desafios

## Algortimos

### Tabela comparativa de modelos (Coeficiente de Determinação — R²)

| Algoritmo        | Coef. de Determinação (Treino) | Coef. de Determinação (Teste) | Coef. de Determinação Médio |
|------------------|-------------------------------|-------------------------------|-----------------------------|
| XGBoost          | **92.90% ✅**| 84.38%| **82.97% ✅**|
| Random Forest    | 91.59%| **84.48% ✅**| 82.83%|
| LGBM             | 88.15%| 82.18%| 82.26%|
| SVR              | 87.15%| 81.88%| 82.37%|
| Árvore Decisão   | 86.76%| 81.39%| 76.15%|


## Desafio
### Tabela comparativa de modelos para o desafio 

|Algortimo|Score de treino|Score de teste|Erro absoluto|Erro absoluto médio|Erro quadratico médio|Raíz do erro quadratico médio| Validação cruzada|
|-|-|-|-|-|-|-|-|
|XGBOOST|88.28%|88.42%|2560.47|2560.47|18462584.28|4296.81|85.18%|
|RANDOM FOREST|88.65%✅|88.97%✅|2482.14✅|2482.14✅|17587869.97✅|4193.78✅|85.57%✅|

### Modelo usado
`RandomForestRegressor(max_depth=5, n_estimators=150, criterion='squared_error', random_state=10)`

### Previsores

Não foram escalonados, apenas foram criadas "variáveis dummy" usando OneHotEncoder e ColumnTransformer, transformando
os valores possíveis da coluna 'region' em colunas, para evitar priorização de valores pelo algortimo.


