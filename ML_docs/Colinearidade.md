# 📌Colinearidade

## ✅ O que é?
Colinearidade acontece quando duas ou mais variáveis independentes (também chamadas de features ou atributos) em um modelo de machine learning ou estatística estão altamente relacionadas entre si.

Ou seja:

➡️ Uma variável pode ser prevista (ou quase prevista) usando outra variável.
Isso causa redundância na informação.


## 🤔 Por que isso é um problema?
Modelos como regressão linear ou logística assumem que as variáveis independentes são realmente independentes.

Se elas forem colineares, o modelo:

* Pode ter dificuldade para estimar os coeficientes corretamente

* Fica menos interpretável

* Pode ficar instável, mudando muito se você trocar um pouco os dados

## 🧠 E com variáveis Dummy?
Quando você transforma uma variável categórica em dummies (ex: a, b, c, d), você cria várias colunas com 0 e 1.

Se você mantiver todas as colunas, acontece isso:

Se a pessoa não é b, c nem d, ela só pode ser a.

Então a coluna a pode ser calculada com:

    a = 1 - (b + c + d)

✅ Isso é colinearidade perfeita.

🧠 Por isso, a gente costuma eliminar uma das colunas dummy, e deixar o modelo descobrir sozinho qual categoria ela representa — isso se chama variável de referência.



## Como resolver?

✅ Se você tem uma variável categórica (como "frequência de corrida")...
* E você tem N pessoas na sua amostra, então:

* Você vai criar uma matriz com N linhas (uma linha por pessoa)

* E vai ter 4 colunas (uma para cada categoria: a, b, c, d)

* Cada linha vai ter um único 1 indicando qual categoria aquela pessoa pertence

* O resto da linha será 0s

### 📌 Exemplo:
Digamos que você tenha essas 4 pessoas:

|Pessoa|Frequência corrida|
|---|---|
|1|a (não corre)|
|2|c (3-4 dias)|
|3|b (1-2 dias)|
|4|d (5-6 dias)|


Você transforma em:

|a|b|c|d|
|---|---|---|---|
|1|0|0|0|
|0|1|0|0|
|0|0|1|0|
|0|0|0|1|

✅ Essa matriz está correta para ser usada como entrada em algoritmos de machine learning.

## 🔁 Resumindo
* Sim! A matriz DEVE representar toda a amostragem, com:

* Uma linha por observação (pessoa)

* Uma coluna por categoria da variável

* Um único 1 por linha, indicando qual valor aquela pessoa tem no atributo