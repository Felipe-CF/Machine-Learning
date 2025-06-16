# Análise dos desempenhos - Algortimos, Desafios

## Algortimos - Multilayer Perceptron

### Tabela comparativa de utilização das redes para regressão e classificação (Coeficiente de Determinação — R²)

`Classificação`
| Algoritmo        | Análise de treino | Coef. de Determinação (Teste) | Coef. de Determinação Médio |
|------------------|-------------------------------|-------------------------------|-----------------------------|
| XGBoost          | **92.90% ✅**| 84.38%| **82.97% ✅**|
| Random Forest    | 91.59%| **84.48% ✅**| 82.83%|
| LGBM             | 88.15%| 82.18%| 82.26%|
| SVR              | 87.15%| 81.88%| 82.37%|
| Árvore Decisão   | 86.76%| 81.39%| 76.15%|

`Regressão`
|Algortimo|Score de treino|Score de teste|Erro absoluto|Erro absoluto médio|Erro quadratico médio|Raíz do erro quadratico médio|Validação cruzada|
|-|-|-|-|-|-|-|-|
|MLP|87.59%|83.19%|452057.16|452057.16|233369297481.05|483083.12|78.52%|



## Desafio

O dataset usado é referente ao disgnóstico de tumores, se maligno (1) ou benigno (0). São utilizados `30 previsores` para gerar a saída prevista na coluna `'diagnosis'`.

### Tipo de rede:  Multilayer Perceptron
![Descrição da rede](https://imgur.com/lf82oZo.jpg)

### Previsores e Modelo usado

rede_neural = MLPClassifier(hidden_layer_sizes=(20), activation='tanh', solver='adam', max_iter=800,
                            tol=.0001, random_state=3, verbose=True, alpha=.0004)

`n° de neurônios por camada oculta` 
    N = (nE + nS) / 2 = (30+1)/2 = 15,5 = 16 
    N = (2 * nE) / 3  + nS = 30*2/3 + 1 = 21 
    nMed = (16 + 21) / 2 = 19 
    nUtilizado = 20

> Ao tentar ampliar a camada oculta para 2, houve uma queda nos resultados para `Acurácia de treino: 62.56%` e `Acurácia de teste: 63.16%`.

### Resultado

    Acurácia de treino: 98.74%

    Acurácia de teste: 97.66%

    Validação cruzada - Acurácia média: 97.36%  


### Previsores

Foram escalonados, mas não foram criadas "variáveis dummy".


