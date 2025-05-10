# Tuplas

Uma tupla é uma estrutura de dados semelhante a uma lista, mas imutável (ou seja, seus elementos não podem ser alterados após a criação).

Caso um dos elementos seja uma `lista` ou um `dict`, ele poderá ser alterado na tupla ( o seu conteúdo)

    minha_tupla = (1, 2, 3)  # Tupla com três elementos

## Características das tuplas
✔ Ordenadas → A ordem dos elementos é mantida.

✔ Podem conter qualquer tipo de dado → Números, strings, listas, etc.

✔ Imutáveis → Não podem ser alteradas após a criação.

✔ Mais rápidas que listas → Menos consumo de memória.

## Lista x Tupla

É possível converter uma lista em uma tupla e vice-versa, utilizando `list()`e `tuple()`.

Enquanto a lsita pode ter seu elementos diretos mudados, `a tupla não`. Essa imutabilidade pode ser usada para `integridade de dados`, para ser usada como `chave:valor` em um dicionário, ou pela velocidade, pois leva vantagem em relação as listas.

## Slicing, Acesso

É semelhante a uma lista, mas lembrando que a tupla é **imutável**

## Métodos úteis

|Método|Descrição|
|---|---|
|count(valor)|	Conta quantas vezes valor aparece
|index(valor)|	Retorna o índice da primeira ocorrência de valor

## Concatenação, Repetição

Igual como nas `listas`

    tupla1 = (1, 2)
    tupla2 = (3, 4)

    print(tupla1 + tupla2)  # (1, 2, 3, 4) -> Concatenação
    print(tupla1 * 3)  # (1, 2, 1, 2, 1, 2) -> Repetição


## Iteração 

Igual como nas `listas`

    tupla = (10, 20, 30)

    for item in tupla:
        print(item)

## Desempacotamento de Tuplas

    tupla = (1, 2, 3)

    a, b, c = tupla  # Atribui os valores da tupla às variáveis
    print(a, b, c)  # 1 2 3

Desempacotando com `resto`

    tupla = (1, 2, 3, 4, 5)

    a, *meio, b = tupla
    print(a)  # 1
    print(meio)  # [2, 3, 4]  (valores intermediários ficam numa lista)
    print(b)  # 5


## Funções Úteis

    tupla = (4, 1, 7, 3)

    print(len(tupla))  # 4 -> Número de elementos
    print(min(tupla))  # 1 -> Menor valor
    print(max(tupla))  # 7 -> Maior valor
    print(sum(tupla))  # 15 -> Soma dos elementos

## Comparações

Exemplo 1

    t1 = (1, 2, 3)
    t2 = (1, 2, 4)

    print(t1 == t2)  # False (1,2,3 ≠ 1,2,4)
    print(t1 < t2)   # True  (3 < 4)
    print(t1 > t2)   # False

> Como 1 == 1 e 2 == 2, o desempate ocorre no terceiro elemento (3 < 4), então t1 < t2 é True.

Exemplo 2 - tuplas com tamanhos diferentes

    t1 = (1, 2)
    t2 = (1, 2, 3)

    print(t1 < t2)  # True (porque t1 "termina" antes de t2)

Exemplo 3 - ordenação

    pessoas = [("Alice", 25), ("Bob", 20), ("Carol", 30)]
    pessoas.sort()  # Ordena primeiro pelo nome

    print(pessoas)  
    # [('Alice', 25), ('Bob', 20), ('Carol', 30)]

    pessoas.sort(key=lambda x: x[1])  # Ordena pela idade
    print(pessoas)  
    # [('Bob', 20), ('Alice', 25), ('Carol', 30)]






## Aplicações 

✅ Retornar múltiplos valores em funções

✅ Usar como chaves em dicionários

✅ Armazenar dados fixos (como meses do ano)

✅ Troca de valores eficiente

✅ Iteração otimizada com enumerate()

✅ Evitar modificações em parâmetros de função

✅ Trabalhar com posições fixas (ex: coordenadas)

✅ Criar registros de dados imutáveis
