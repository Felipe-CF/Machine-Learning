# Regressão  - Correlação Linear, Regressão Linear Simples, Regressão Linear Múltipla, Máquina de vetor de suporte (SVM),  Árvores de decisão (decision tree), Random Forest, XGBOOST, Light GBM (Light Gradient Boosting Machine), CatBoost (category boosting), Redes Neurais Artificiais

## O que é?

Utiliza os dados de entrada para treinar um algoritmo de ML e, assim, prever uma resposta numérica.

## Dataset utilizado

O *Boston housing data* foi criado em 1978, cada um das suas 506 entradas representam dados agregados sobre 14 características de casas de vários suburbos de Boston, Massachusetts.

## Correlação Linear

É a relação linear entre dias variáveis, determinado através de gráficos de dispersão e do coeficiente de correlação.

### **OBS**: *correlação não é causalidade*  

* positiva: x > 0
* negativa: x < 0

![Gráfico de correlações](https://imgur.com/cTsgPnM.png)

### Coeficiente de correlação 
![Gráfico de correlações](https://imgur.com/k6DQAhg.png)

| Valor de `+r`  | Interpretação Positiva       | Valor de `-r`  | Interpretação Negativa       |
|------------------------|-------------------------------|------------------------|-------------------------------|
| `+1.0`                 | Correlação perfeita positiva  | `-1.0`                 | Correlação perfeita negativa  |
| `+0.7` a `+0.9`        | Correlação forte positiva     | `-0.7` a `-0.9`        | Correlação forte negativa     |
| `+0.4` a `+0.6`        | Correlação moderada positiva  | `-0.4` a `-0.6`        | Correlação moderada negativa  |
| `+0.1` a `+0.3`        | Correlação fraca positiva     | `-0.1` a `-0.3`        | Correlação fraca negativa     |
| `0`                   | Sem correlação                | `0`                   | Sem correlação                |

### Coeficiente de determinação (r²)

Porcentagem da variação de *y* que pode ser explicada pela relação de *x e y*. Avalia a qualidade do ajuste de um modelo de regressão (o quanto ele se encaixa aos dados).

`r²` = coeficiente de correlação ao quadrado



## Regressão Linear Simples

Utilizado posteriormente à análise de correlação linear.

**Fórmula**
`y = mx + b`

A equação é obtida após um ajuste de uma reta no gráfico de disperção com resíduo mínimo, isso é feito com a **linha de regressão**, que contém valore teóricos próximo aos plotados no gráfico.

`Linha de regressão`
É aquela que melhor se ajusta aos dados plotados, onde **a soma dos quadrados dos resíduos seja mínima**. Assim temos:

> para um dado x, temos d = (valor y observado) - (valor y previsto)

`Modelo matemático`

* y = mx + b

    n * somat(xy) - somat(x) * somat(xy) 
m = ---------------------------------------
    n + somat(x²) - (somat(x)²)

b = media(y) - m*media(x)

    somat(y)          somat(x)
b = ---------  -  m * --------
        n               n


## Regressão Linear Múltipla

## Máquina de vetor de suporte (SVM)

## Árvores de decisão (decision tree)

## Random Forest

## XGBOOST

## Light GBM (Light Gradient Boosting Machine)

## CatBoost (category boosting)

## , Redes Neurais Artificiais

**Seleção de características**: seleciona os melhores atributos e utiliza sem transformações

**Extração de características**: encontra os relacionamentos dos melhores atributos e cria novos



