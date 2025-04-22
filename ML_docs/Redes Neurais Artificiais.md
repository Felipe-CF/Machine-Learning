# Redes Neurais Artificiais

## Neurônio artificial

Composto por sinais de entrada com *pesos sinápticos*, onde enviam sinais para uma *combinação linear*. Depois esse resultado é enviado para uma *função de ativação*, para então ser gerado  uma saída.

## A Rede neural

`conjunto de neurônios interligados`
### Construção

1) Função de ativação

2) Algoritmo de aprendizagem

3) Topologia da rede

### Camadas

`Entrada`: input
> cada coluna/atributos do dataset representa 1 neurônio nessa camada

`Oculta`: hidden layer

`Saída`: output

`apenas 1 camada oculta`: **rede neural simples**

`mais de 1 camada oculta`: **rede neural profunda** (deep learning)

## Perceptron (1 camada)

Algoritmo mais simples, capaz de classificar padrões que sejam **linearmente separáveis**. Tem *+1 entrada* e *1 saída binária*.

`Funcionamento`: combinador linear irá somar cada entrada com seu respectivo peso, depois as função de ativação ditará a saída binária:

    1 --> w * x + b >= 0

    0 --> w * x + b < 0

`Bias`(viés): aumenta o grau de liberdade dos ajustes dos pesos


### Rede Multilayer Perceptron (camada oculta)
Todos os neurônios são ligados **somente** aos da câmada subsequente (sem ligação lateral), sendo usados para padrões **não linearmente separáveis**.
> o principal algoritmo de treinamento é o **error backpropagation**.
## Regras de Aprendizagem

### Correção de erro

Ajusta os pesos por meio do **erro**, que é obtido pela **diferença do valor de saída da rede e o esperado pelo ciclo de treinamento**. A diminuição do erro pe gradual.

### Hebbiana

`Postulado`: Se 2 neurônios, em ambos os lados de uma sinapse, **são ativados sincrona e simultaneamente**, então a **força daquela sinapse é seletivamente aumentada**.

### Competição

Neurônios forçados a competirem, pois somente 1 será ativado, sendo o vencedor **o de maior similaridade com o padrão de entrada**. Os pesos dos neurônios próximos a ele terão seus valores ajustados.

### Descida do gradiente

Gradiente é o vetor cujo módulo é a **derivada direcional máxima** (sentido da maior variação). Ele aponta para onde a grandeza resultado da função tem maior crescimento, sendo utilizado para encontrar ou se paroximar do **mínimo de uma função de erro** (mínmo global)

## Topologia das redes neurais

`Feed forward (alimentada adiante)`  
* forma de camadas    
* neurônios em conjuntos distintos e ordenados sequencialmente    
* fluxo da *camada de entrada* para *a de saída*   

`Feed backward (alimentação recorrente)`

*  ocorrência de realimentação da *camada de entrada*

`Competitivas`

* os neurônios se dividem em 2 camadas: *entrada* (fonte) e *saída* (grade)
* os da grade são forçados *a competir entre si* e *somente o vencedor é ativado*.


## Erro backpropagation

Segue o fluxo inverso do forward propagation, pois se o resultado não for satisfatório, ele *retorna para ajustar os pesos*. Ele **aumenta a descida do gradiente**, sendo imprescindível para o **deep learning**.


## Definição dos  Hiperparâmetros

|Hiperparâmetros|Soluções, Objetivos e Considerações||||||
|-|-|-|-|-|-|-|
|N° de camadas ocultas| normalmente 2 já resolvem ||||||
|N° de neurônios das camadas ocultas| N = (nE + nS) / 2 |N = (2 * nE) / 3 + nS |em excesso = **overfitting**|em escassez = **underfitting**|||
|Taxa de aprendizagem| varia de *0.1 a 1* (sugestão: 0.4)|muito baixo=aprendizado lento|muito alta: oscila o treinamento||||
|Momento| varia de **0** (não utilizaçao) **a 1** (sugestão: 0.3)|aumento da velocidade de treino da rede|redução do perigo de instabilidade (ficar preso em minimos locais)||||
|Parada de treinamento por n° de ciclos| varia de **500 a 3000** ciclos (sugestão)|n° de vezes que o conjunto de treinamento é utilizado pela rede|em excesso = **overfitting**|em escassez = **underfitting**|||
|Parada de treinamento por erro|  **0.1** (sugestão inicial sendo ajustado em função do resultado)|encerra o treino após o **erro quadrático médio** ficar abaixo do valor pré definido|||||
