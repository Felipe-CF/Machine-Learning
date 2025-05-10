# Funções, Args x Kwargs, Lambda, Documentação de funções


## Funções built-in

`input()` entrada de valores

```
 i = input()

 i = type(input())

 complexos d = 42 + 3j

 decimais d = 42.0
```

`random()` valores 'pseudorandomicos'

```
 import random

 r = random.randrange(i, f) 
 
 r = random.random()  --> número decimal aleatório entre 0 e 1

 r = random.randint()  --> número inteiro aleatório

 letras = ['a', 'b', 'c']
 random.choice(letras)  --> indice aleatório da lista

 random.shuffle(letras)  --> embaralho aleatório da lista

 random.uniform(5.5, 9.5)  --> gera ponto flutuante aleatório entre os valores

```

`while`

```
while (condicao):

else:
    fluxo
```

`for`

```
for i in range(inicio, fim,  passo):
    # fluxo

for item in lista:
    # fluxo

```

`map()` aplica uma função a cada elemento iteravel e retorna um iterador

`list()` retorna uma lista

```
#crio uma lista com valores lidos e separados por espaços em branco
lista = list(map(int, input("digite os inteiros).split()))
```

`filter(função, sequencia)`  filtrar elementos de uma lista com base em uma função condicional

```
numeros = [1, 2, 3, 4, 5, 6]

pares = list(filter(lambda x: x % 2 == 0, numeros))

print(pares)  # Saída: [2, 4, 6]
```

`reduce(função, sequencia, valor_inicial (opcional) )`  serve para acumular valores de uma lista, aplicando uma função cumulativa de dois em dois elementos.

```
from functools import reduce

numeros = [1, 2, 3, 4, 5]

soma = reduce(lambda x, y: x + y, numeros)

print(soma)  # Saída: 15

Primeiro: 1 + 2 = 3 
Depois: 3 + 3 = 6
Depois: 6 + 4 = 10
Depois: 10 + 5 = 15

```


`funções simples`

```
def func(param)

def func(param='default') --> se não for passado, considera default  

def func(*args) --> vários argumentos recebidos como uma tupla

```


## Kwargs

### *agrs x **kwargs

O *args recebe variáveis posicionais, enquanto que **kwargs recebe argumentos nomeados (chave=valor) 

> **O `**kwargs` é um dict e o nome em si pode ser mudado, sendo obrigatório `**`**


```
def func(**kwargs):
    if "nome" in kwargs:
        print(f"Nome: {kwargs['nome']}")
    if "idade" in kwargs:
        print(f"Idade: {kwargs['idade']}")

func(nome="Carlos", idade=30)
```

>iterando sobre o kwargs

```
def func(**kwargs):
    for chave, valor in kwargs.items():
        print(valor)
```

>iterando sobre o args

```
for num in args[i:f]
    # fluxo
```

## Funções como argumentos

```
def saudacao(nome):
    return nome

def cumprimentar(funcao, nome)
    return funcao(nome)

print(cumprimentar(saudacao, 'eu'))
```

## Lambda

**lambda argumentos: expressão**


Ao contrário de funções `def`, ela pode ter apenas uma expressão e retorna implicitamente o resultado. Ideal para operaçãos onde não se quer nomear uma função completa.

> dobrar um numero

```
dobrar = lambda n: n * 2
print(dobrar(5))
```

> ordenar uma list

```
dados = [(1, "banana"), (3, "maçã"), (2, "uva")]

dados.sort(key=lambda item: item[1]) 

print(dados)

# Saída: [(1, 'banana'), (3, 'maçã'), (2, 'uva')]
```


> usando map() para retornar uma lista com os resultados

```
numeros = [1, 2, 3, 4]
dobrados = list(map(lambda x: x * 2, numeros))
print(dobrados)  # Saída: [2, 4, 6, 8]
```
***Não é possível alterar uma variável global!***


## Documentação de funções

```
def somar(a, b):

"""
documentação
"""

    return a + b

print(somar.__doc__) # imprime a docstring da função
```