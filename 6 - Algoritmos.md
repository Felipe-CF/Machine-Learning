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


