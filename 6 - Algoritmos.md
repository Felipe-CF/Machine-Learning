# Naive Bayes, Máquina de vetor de suporte (SVM), Regressão logística, Aprendizagem baseada em instâncias (KNN), Árvores de decisão (decision tree), Random Forest

> Resultados usando a base heart_tratado.csv

Bayes = 
SVM =
Regressão logística =
KNN = Acurácia: 89.39% , Acertos: 276, 

## Naive Bayes 

CLassificador probabilístico baseado no teorema de Bayes.


![Formula](https://imgur.com/kCo5g2Y.jpg)


**Premissa**: independência entre as variáveis

**Obs**: trabalha bem com variáveis categóricas

### Vantagens
É de fácil entendimento, pouco esforço computacional, bom desempenho com muitos dados e previsões com poucos dados.

### Desvantagens
Consiedrar atributos independentes, **atribui nula probabilidade** quando uma classe contida no teste não se apresente no treino.


### Código

    x_train, x_test, y_train, y_test = train_test_split(previsores3, alvo, random_state=0, test_size=0.3) 

    naive = GaussianNB()

    naive.fit(x_test, y_test)

    previsoes_teste = naive.predict(x_test)

***arquivo: bayes.py***


## Máquina de vetor de suporte (SVM)

Aplicado em problemas de aprendizagem supervisionada tanto de classificação como de regressão. **Em classificaçao** é conhecido como ***CLassificador de Vetor de Suporte*** (SVC). O objetivo é **criar hiperplanos de separação dos dados**, podendo ser aplicado em *problemas linearmente e não linearmente separáveis*.

![SVM](https://imgur.com/ejRktq9.jpg)

### Equação

> w * x + b >= 0

* w: 
* x: 
* b: 

### Aplicações

* Classificação
* Categorização de textos
* Reconhecimento de imagem e letras manuscritas
* Detecção facial e de anomalias

### Constante de penalização (custo)

> Os **hiperparâmetros** são aqueles que é possível alterar/ajustar para conseguir um melhor resultado no algortimo. 

**Hiperparâmetro C**: controla a tolerância dos erros.

Quanto **maior o valor de C**, maior o poder de separação das classes, porém maior a probabilidade de *overfitting* e do *tempo de treinamento*. Quão **menor for o valor de C**, aumenta a chance de erros na sepração e *underfitting*.

O hiperparâmetro **gamma** pode ser ajustado para otimização.

### Vantagens

* não é influenciado por dados discrepantes
* solução de problemas lineares e não lineares
* efetivo em datasets grandes
* consegue aprender com características não pertencentes aos dados

### Desvantagens

* difícil visualização gráfica e interpretação teórica
* mais lento em comparação a outros
* cuidado com os hiperparâmetros para evitar *overfitting* e *underfitting*


### código

'''
from sklearn.svm import SVC

svm = SVC(kernel='rbf', random_state=1, C=2)

svm.fit(x_train, y_train)
'''

## Regressão logística

Recebe esse nome devido a utilização de conceitos de **regressão linear** em seu modelo matemático. Ela pode ser **usada em problemas binários**, variável dependente binária (duas saídas),  **ou multinomila** (variável dependente com mais de duas categorias)

![Formula](https://imgur.com/KxTBsV7.jpg)

* p= probabilidade de pertencer a determinada classe
* e= número de Euler
* b0= intercepto
* bn= coeficientes
* xn= variáveis dependentes

### Comportamento

A **regressão linear** tende a extrapolar os limites pois: 𝑥 → ±∞,  y → ±∞. Isso é problemático quando se trata de probabilidades (entre 0 e 1). Já a **logística** limita a saída a: 0 < y < 1. Ela é ideal para **prever probabilidade de eventos binários** (sim ou não).

![Comparação entre as regressões](https://imgur.com/OB1uQ53.jpg)

O **limiar de decisão** é usado para escolher/definir o resultado, pois caso o **valor esteja acima**, a **saída será 1**, **se for menor será 0**.

### Vantagens

* fácil implementação
* teoria consolidada
* excelente desempenho
* indica o valor de probabilidade para cada instância 

### Código e Parâmetros

* C = controla a tolerância dos erros (separa as classes).
* penalty = evita overfitting, multicolinearidade
* max_iter = evita overfitting e torna o modelo mais simples e generalizável definindo limite de iterações que o algoritmo terá para melhorar o resultado 
* solver = algoritmo que busca otimização e menor erro possível (ligado com o 'penalty')
* max_iter = limite de iterações que o algoritmo terá para melhorar o resultado
* tol = trabalha junto com 'max_iter' definindo o limite de erros


## Aprendizagem baseada em instâncias (KNN)

Define agrupamento nos dados e, ao chegar bivis dados, traça uma área ao redor dele para identificar a que grupo ele pertence.

![KNN](https://imgur.com/5TKGFoh.jpg)


## Árvores de decisão (decision tree)

Aplicado em problemas de aprendizagem supervisionada tanto de **classificação (mais utilizado)** como de **regressão**.

Seleciona a ordem que os atributos irão aparecer na árvore, *sempre de cima para baixo*, conforme sua *importância para a predição*, assim como determina a separação dos ramos da árvore.

![Exemplo](https://imgur.com/j0LO8iV.jpg)

### Podagem

Objetiva diminuir a probabilidade de overfitting.
Pode ser de duas formas:

1) Pré-podagem: parar o crescimento da árvore.
2) Pós-podagem: poda com a árvore já completa.

**Processo de podagem**:
- Percorre a árvore em profundidade.
- Para cada nó de decisão calcula o erro no nó e a soma dos erros nos nós descendentes.
- Se o erro do nó é menor ou igual à soma dos erros dos
nós descendentes então o nó é transformado em folha.



## Random Forest

Criação aleatória de várias árvores de decisão.

Utiliza o método Ensemble (construção de vários modelos para obter um resultado único).

É mais robusto, complexo e normalmente propicia resultados melhores, mas possui maior custo computacional.

Em *problemas de classificação* o resultado que *mais aparece* será o escolhido (moda), já em **regressão será a média**.

![Ilustração](https://imgur.com/K41BnRM.jpg)

### Diferenças referente a Árvore de decisão

|Árvore de decisão|Random Forest|
|---|---|
|apenas uma árvore|conjunto de árvores.|
|cria regras para seleção das melhores variáveis|seleção das variáveis aleatoriamente.|
|resultado é “fruto” de uma única árvore|resultado é a moda ou média de todas as árvores.|

### Vantagens X Desvantagens

|Vantagens|Desvantagens|
|---|---|
|Resultados bastante precisos.|Velocidade de processamento relativamente baixa|
|Normalmente não necessitam de preparações sofisticadas nos dados (label Encoder e OneHot Encoder)|Difícil interpretação de como chegou no resultado|
|Trabalha com valores faltantes, variáveis categóricas e numéricas.|---|
|Pouca probabilidade de ocorrência de overfitting.|---|


