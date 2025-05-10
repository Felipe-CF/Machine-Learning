# **Variavéis e Strings**

## Variavéis

```

 inteiro d = 42

 complexos d = 42 + 3j

 decimais d = 42.0

```

### global e local

```
x = 10  # Variável global

def alterar():
    x = 20  # Isso cria uma variável local x, sem afetar a global
    print(x)  # Saída: 20 (mas a global continua 10)

alterar()
print(x)  # Saída: 10 (a variável global não foi alterada)

```
> modificando o valor da variável global

```
x = 10  # Variável global

def alterar():
    global x  # Indica que x é a variável global
    x = 20  # Modifica o valor de x

alterar()
print(x)  # Saída: 20

```

**se mais de um arquivo .py precisar de uma variável global, posso definir em um outro .py e importar e acessar ela**

### non local

É usada em escopos mais próximos ou superioress, em funções aninhadas

```
def func1():
    var = 'externa'

    def func2():
        nonlocal var

        var = 'interna'

        print(var)
```

### variáveis de funções

```
def func():
    return 'ok'

teste = func

print(teste())
```


## Strings

`Slice`

```
string[i:f] recorta o str entre o indice 'i' e 'f'

string[-n:] n último caracteres
```

`len(string)` tamanho do str

`string.strip()` remove espaços em brancos por padrão, mas pode receber um caracter 

`lista = string.split()` separa em lista, com espaços sendo separadores

`string = ''.join(lista)` une em lista, com espaços sendo separadores

`string = string.upper()` maiusculo

`string = string.lower()` minusculo

`string = string.capitalize()` primeiro caractere maiusculo

`string = string.count()` conta ocorrencia de caractere 

`string = string.replace("string", 'string')` troca uma string por outra (se existir) 

`Formatação`

```
mensagem = f"teste {i}"

mensagem = f"teste {i:.2f}" duas casas decimais

mensagem = "teste {} {}.format(i, f)" 

```