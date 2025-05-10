# ***Conjuntos ou Sets***

## O que é?

Conjunto desordenado de elementos únicos, sendo semelhantes aos conjuntos numéricos já conhecidos. Os elementos de um set são **imutáveis**, mas o **próprio set é mutável**.

Isso não permite que `listas` ou outros `sets` sejam elementos de um set, pois eles em sim são mutáveis. Podemos `ter`, `str`, `int`, `float` ou `tupla`.

## Criação

`Usando chaves {}`

    meu_set = {1, 2, 3, 4}

`Usando a função set()`

    outro_set = set([1, 2, 3, 3, 4, 4])  # Remove duplicatas automaticamente

**Criação de conjunto vazio**
>Atenção! Para criar um set vazio, use **set()**, pois {} cria um dicionário.

## Operações

>Adição

    meu_set.add(5)

> Remoção

    meu_set.remove(2)  # Remove o 2 (gera erro se não existir)
    
    meu_set.discard(3) # Remove o 3 (não gera erro se não existir)

    meu_set.pop()      # Remove um elemento aleatório

> Busca

    print(4 in meu_set)

> União

    set1 = {1, 2, 3}

    set2 = {3, 4, 5}
    
    uniao = set1 | set2  # {1, 2, 3, 4, 5}

> Intersecção

    set1 = {1, 2, 3}

    set2 = {3, 4, 5}

    inter = set1 & set2  # {3}

> Diferença

    set1 = {1, 2, 3}

    set2 = {3, 4, 5}
    
    d = set1 - set2  # {1, 2}

> Diferença simétrica (elementos que estão em apenas um dos sets)

    dif_sim = set1 ^ set2 # {1, 2, 4, 5}

    set1.symmetric_difference(set2) # {1, 2, 4, 5}

> Subconjunto

    set1.issubset(set2)

> Superconjunto

    set2.issuperset(set1)

## Outras funções e métodos

`copy()` 

Copia a referencia do set para uma outra variavel

## Imutabilidade e frozenset

Um frozenset é uma versão imutável do set.

🔹 Diferença principal:

> set → Mutável (podemos adicionar e remover elementos).

> frozenset → Imutável (não podemos modificar depois de criar)

    fs = frozenset([1, 2, 3, 4])

> um `frozenset` **PODE SER USADO COMO ELEMENTO EM UM SET** 


## Limitações

Conjuntos `não suportam indexação`, `fatiamento` (não são sequencias ordenadas) ou outras `operações de sequencia`. Também não pode conter elementos duplicados.

## Quando usar?

✅ Quando você precisa armazenar elementos únicos.

✅ Quando precisa de operações rápidas de busca (in é mais rápido que em listas).

✅ Quando precisa comparar conjuntos de dados (união, interseção, diferença).



