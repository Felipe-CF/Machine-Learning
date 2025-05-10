# Listas


## 📌 Métodos de Operações

`list(iterável)` → Cria uma lista a partir de um iterável.

> o ***list*** pode criar a partir de uma ***tupla*** ou um ***set***, porém no caso deste a ordem pode acabar mudando

`append(x)` → Adiciona x ao final da lista.

`extend(iterável)` → Adiciona os elementos de um iterável à lista.

`insert(i, x)` → Insere x na posição i.

> Remoção 

`remove(x)` → Remove a primeira ocorrência de x.

`pop(i)` → Remove e retorna o elemento na posição i (ou o último, se i não for informado)

`del lista[i]` → Remove um elemento pelo índice

`clear()` → Remove todos os elementos da lista

> Ordenação e contagem
---

`sort(reverse=False, key=...)` → Ordena a lista (pode usar key para critérios personalizados).

`reverse()` → Inverte a ordem dos elementos.

`count(x)` → Conta quantas vezes x aparece na lista.

`index(x, start=0, end=len(lista))` → Retorna o índice da primeira ocorrência de x.

> Cópia e duplicação
---

`copy()`→ copia a lista

## Entendo copy() e copy.deepcopy()

> **copy()** Compartilham a referência (alterações afetam ambas), já **deepcopy()**, Cada elemento é copiado de forma independente. Elementos como lista são passados por referência, por isso é necessário usar a segunda opção.

> Use **deepcopy()** quando sua lista contém objetos mutáveis aninhados e você precisa de uma cópia totalmente independente.


📌 **Extras úteis**:


`sum(lista)` → Retorna a soma dos elementos (se forem numéricos).

`min(lista)` / `max(lista)` → Retorna o menor/maior valor.

`len(lista)` → Retorna o número de elementos.

`lista = [] * n` → repete os elementos da N vezes (total)


### Entendo o zip()

`zip()` → Ele retorna um iterador de tuplas, onde cada tupla contém um elemento de cada iterável na mesma posição.

    nomes = ["Ana", "Pedro", "João"]
    notas = [8, 10, 7]

    pares = zip(nomes, notas)
    print(list(pares))  # [('Ana', 8), ('Pedro', 10), ('João', 7)]

`zip_longest()`

Quando as listas têm tamanhos diferentes, os elementos extras da maior são **ignorados**. Para lidar com isso, basta usar o zip_longest(), que preenche a tupla com None, ficando **(None, maior_lista_item)**

    from itertools import zip_longest

    nomes = ["Ana", "Pedro"]
    notas = [8, 10, 7]  # Tem um elemento a mais

    pares = zip_longest(nomes, notas)
    print(list(pares))
    # [('Ana', 8), ('Pedro', 10), (None, 7)]

Se houver necessidade, o parametro `fillvalue` permite que você defina um valor padrão para o preenchimento

    pares = zip_longest(nomes, notas, fillvalue="Sem nome")

    print(list(pares))

    # [('Ana', 8), ('Pedro', 10), ('Sem nome', 7)]

Quando se deseja descompactar o zip, basta fazer assim: `lista1, lista2 = zip(*dados)`


## Slicing de listas

`lista[início:fim:passo]`

início → Índice inicial (inclusivo)

fim → Índice final (exclusivo)

passo → De quantos em quantos elementos avançar

> Se algum dos parâmetros for omitido, o Python assume valores padrão:

início = 0 (começa do primeiro elemento)

fim = len(lista) (vai até o final)

passo = 1 (pula de um em um)

    numeros = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    print(numeros[2:6])  # [2, 3, 4, 5] (índice 6 não entra)



## List comprehension

List comprehension é uma forma concisa e eficiente de criar listas em Python, substituindo loops for e funções map().

> nova_lista = [expressão for item in iterável if condição]

    expressão → O que será adicionado à nova lista

    item → Cada elemento do iterável

    iterável → A sequência a ser percorrida

    if condição (opcional) → Filtra os elementos

### 📌 Quadro Resumo

|Sintaxe|Equivalente|
|---|---|
|[x for x in lista]|	for simples
|[x for x in lista if condição]|	for + if
|[x if condição else y for x in lista]|	for + if-else
|[f(x) for x in lista]|	map()
|{chave: valor for x in lista}|	Dicionário

### Exemplos


    combinacoes = [(x, y) for x in range(3) for y in range(2)]

    print(combinacoes)  # [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]

> List Comprehension com zip()

    nomes = ["Ana", "Pedro", "João"]
    notas = [8, 10, 7]

    boletim = {nome: nota for nome, nota in zip(nomes, notas)}

    print(boletim)  # {'Ana': 8, 'Pedro': 10, 'João': 7}


## Enumerate

O `enumerate()` é usado para percorrer uma lista ao mesmo tempo em que retorna o índice e o valor de cada elemento.

    enumerate(iterável, start=0)

### Exemplo
    frutas = ["maçã", "banana", "uva"]

    for indice, fruta in enumerate(frutas):
        print(f"{indice}: {fruta}")

    # Saída:
    # 0: maçã
    # 1: banana
    # 2: uva

É possível converter o enumerate em uma lista, onde cada elemento é um tupla (indice, valor)

    lista_enumerada = list(enumerate(frutas))
    print(lista_enumerada)

    # Saída:
    # [(0, 'maçã'), (1, 'banana'), (2, 'uva')]



