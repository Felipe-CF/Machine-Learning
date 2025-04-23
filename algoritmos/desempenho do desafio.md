# Análise dos desempenhos

## Tabela comparativa

|Algortimo|Score de treino|Score de teste|Erro absoluto|Erro absoluto médio|Erro quadratico médio|Raíz do erro quadratico médio| Validação cruzada|
|-|-|-|-|-|-|-|-|
|XGBOOST|88.28%|88.42%|2560.47|2560.47|18462584.28|4296.81|85.18%|
|RANDOM FOREST|88.65%✅|88.97%✅|2482.14✅|2482.14✅|17587869.97✅|4193.78✅|85.57%✅|

## Modelo usado
`RandomForestRegressor(max_depth=5, n_estimators=150, criterion='squared_error', random_state=10)`

## Previsores

Não foram escalonados, apenas foram criadas "variáveis dummy" usando OneHotEncoder e ColumnTransformer, transformando
os valores possíveis da coluna 'region' em colunas, para evitar priorização de valores pelo algortimo.


