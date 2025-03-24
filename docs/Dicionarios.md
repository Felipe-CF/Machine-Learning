# Dicionários

Os dicionários (dict) são estruturas de dados chave-valor. Eles são mutáveis, não ordenados (até Python 3.6) e permitem acesso rápido aos valores por meio das chaves.


    dados = {chave: valor}

## Acessando valores

    print(dados["nome"])   # Alice

`dict["chave"]` → Retorna o valor, gera erro se a chave não existir.

    print(dados.get("idade"))  # 25 (get evita erro se a chave não existir)

`dict.get("chave", valor_padrão)` → Retorna o valor, não gera erro se a chave não existir.

    print(dados.get("profissao", "Desconhecido"))  # "Desconhecido" (chave não existe)

## Manipulando dados

    dados["profissao"] = "Engenheira"  # Adiciona nova chave, caso não exista

    dados["idade"] = 26  # Modifica um valor existente

    print(dados)  
    # {'nome': 'Alice', 'idade': 26, 'cidade': 'São Paulo', 'profissao': 'Engenheira'}

## Removendo dados

    del dados["cidade"]  # Remove a chave 'cidade'

    profissao = dados.pop("profissao")  # Remove e retorna o valor da chave

    dados.clear()  # Remove todos os itens
    print(dados)  # {}

## Iterando sobre o dict

    dados = {"nome": "Alice", "idade": 25, "cidade": "São Paulo"}

    for chave in dados:  
        print(chave)  # nome, idade, cidade

    for valor in dados.values():
        print(valor)  # Alice, 25, São Paulo

    for chave, valor in dados.items():
        print(f"{chave}: {valor}") 

    # nome: Alice
    # idade: 25
    # cidade: São Paulo


## Métodos úteis

|Método|Descrição|
|---|---|
|dict.keys()|	Retorna todas as chaves|
|dict.values()|	Retorna todos os valores|
|dict.items()|	Retorna uma lista de tuplas (chave, valor)|
|dict.update(outro_dict)|	Atualiza o dicionário com outro dicionário|
|dict.pop("chave")|	Remove e retorna o valor da chave|
|dict.clear()|	Remove todos os itens|
|dict.setdefault()|	retorna o valor da chave, mas se não existir insere a chave com valor especificado|
|***dict***.fromkeys()|Crio um dicionário com as chaves passadas recebendo o valor especificado na função|

    print(dados.keys())   # dict_keys(['nome', 'idade', 'cidade'])

    print(dados.values()) # dict_values(['Alice', 25, 'São Paulo'])

    print(dados.items())  # dict_items([('nome', 'Alice'), ('idade', 25), ('cidade', 'São Paulo')])


### dict.update()

O método .update() `permite atualizar um dicionário com os pares chave-valor de outro dicionário ou de um iterável de pares (tuplas)`. Se a chave já existir, o valor será substituído; se não existir, será adicionado.

Exemplo

    dados = {"nome": "Alice", "idade": 25}

    novos_dados = {"cidade": "São Paulo", "idade": 26}

    dados.update(novos_dados)

    print(dados)

    # {'nome': 'Alice', 'idade': 26, 'cidade': 'São Paulo'}

Os dados podem ser passado como uma `lista de tuplas(chave, valor)`, `um dict`, usando `argumentos nomeados (kwargs)`(abaixo). Se o valor exsitir, ele é alterado, se não, adicionado ao dict.

    dados = {"nome": "Alice"}

    dados.update(idade=25, cidade="São Paulo")

    print(dados)

    # {'nome': 'Alice', 'idade': 25, 'cidade': 'São Paulo'}



## Dict Comprehension

O dict comprehension permite criar dicionários de forma concisa e eficiente, semelhante ao list comprehension, mas usando chaves e valores.

    {chave: valor for item in iterável}

Exemplo 1 - invverter chave:valor

    dicionario = {"a": 1, "b": 2, "c": 3}

    invertido = {v: k for k, v in dicionario.items()}

    print(invertido)

    # {1: 'a', 2: 'b', 3: 'c'}

Exemplo 2 - criando um dict

    lista = [("nome", "Alice"), ("idade", 25), ("cidade", "São Paulo")]

    dicionario = {chave: valor for chave, valor in lista}

    print(dicionario)
    
    # {'nome': 'Alice', 'idade': 25, 'cidade': 'São Paulo'}




## Dicionários Aninhados

    usuarios = {
        "user1": {"nome": "Alice", "idade": 25},
        "user2": {"nome": "Bob", "idade": 30}
    }

    print(usuarios["user1"]["nome"])  # Alice


## Aplicações práticas
📌 Dicionários são úteis para: 

✅ Representar objetos do mundo real (ex: um usuário, produto)

✅ Armazenar configurações de um programa

✅ Criar tabelas de mapeamento (ex: código → significado)

✅ Trabalhar com dados estruturados, como JSON
