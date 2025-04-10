# Naive Bayes, Máquina de vetor de suporte (SVM)


## Naive Bayes 

CLassificador probabilístico baseado no teorema de Bayes.


![Formula](https://imgur.com/kCo5g2Y.jpg)


**Premissa**: independência entre as variáveis

**Obs**: trabalha bem com variáveis categóricas

### Vantagens
É de fácil entendimento, pouco esforço computacional, bom desempenho com muitos dados e previsões com poucos dados.

### Desvantagens
Consiedrar atributos independentes, **atribui nula probabilidade** quando uma classe contida no teste não se apresente no treino.


## Código

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




