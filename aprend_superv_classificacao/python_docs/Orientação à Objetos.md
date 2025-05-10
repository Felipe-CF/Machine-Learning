# Orientação à Objetos

## Conceitos básicos


## Classes e Objetos

Uma classe é um modelo para criar objetos. Um objeto é uma instância de uma classe.


    class Pessoa:
    def __init__(self, nome, idade):  # Construtor
        self.nome = nome  # Atributo
        self.idade = idade

    def saudacao(self):  # Método
        return f"Olá, meu nome é {self.nome} e tenho {self.idade} anos."

    # Criando um objeto (instância da classe)
    p = Pessoa("Ana", 25)
    print(p.saudacao())  # Olá, meu nome é Ana e tenho 25 anos.

`self`serve para fazer referência a instância do objeto atual

`__init__` é o construtor da classe

## Encapsulamento

Controla o acesso aos atributos e métodos usando modificadores:
|Acesso|Método|Descrição|
|---|---|---|
|Público|(self.atributo)|Acesso livre.|
|Protegido|(self._atributo)|Uso interno da classe e subclasses.|
|Privado|(self.__atributo)|Não pode ser acessado diretamente fora da classe.|

>Obs: para acessar um atributo privado pode ser usado o `name mangling` que é **instancia_Classe__atributo**.


## Propriedades

### Decorador **@property**

Em python, podemos usar o decorador @property para ***criar uma propriedade que atua como um atributo***, mas que na verdade é acessada através de um método. Isso é útil para executar alguma lógica extra ao obter ou definir o valor de atributo, permitindo acessar um método como se fosse um atributo, sem precisar de parênteses ().

***Porque usar?***

>Encapsulamento: Permite controlar o acesso a atributos privados.

>Segurança: Evita a modificação direta de atributos.

>Flexibilidade: Permite adicionar lógica ao acesso de atributos sem mudar a interface da classe.

Exemplo:

    class Pessoa:
        def __init__(self, nome):
            self._nome = nome  # Atributo "protegido" (por convenção)

        @property
        def nome(self):  # Método que age como um atributo
            return self._nome

    p = Pessoa("Ana")

    print(p.nome)  # Ana


***@<metodo_getter>.setter***

Permite que a lógica do property seja utilizada no `set` e `delete`, utilizando o mesmo nome para ambas as funções de `get` e `set`.

    @nome.setter
    def nome(self, novo_nome):  # Setter
        if isinstance(novo_nome, str) and novo_nome.strip():
            self._nome = novo_nome
        else:
            raise ValueError("Nome inválido!")

    p = Pessoa("Ana")
    p.nome = "Carlos"  # Chama o setter
    print(p.nome)  # Carlos

***@<metodo_getter>.deleter***

    @nome.deleter
    def nome(self):
        print("Nome removido!")
        self._nome = None  # Ou pode remover completamente com `del self._nome`

    p = Pessoa("Ana")
    del p.nome  # Chama o deleter
    print(p.nome)  # None


## Herança

Uma classe pode herdar atributos e métodos de uma outra, sendo possíveis `sobrescrever` ou `acessar` esses ***métodos pai*** para a classe qeu herda.

    class Animal:
        def __init__(self, nome):
            self.nome = nome

        def fazer_som(self):
            return "Som genérico"

    class Cachorro(Animal):  # Herdando de Animal
        def fazer_som(self):  # Sobrescrevendo método
            return "Latido"

    dog = Cachorro("Rex")
    print(dog.nome)  # Rex (herdado de Animal)
    print(dog.fazer_som())  # Latido (método sobrescrito)

Ao user `super()` podemos acessar métodos da classe pai dentro da classe filha.

    class Gato(Animal):
        def __init__(self, nome, cor):
            super().__init__(nome)  # Chamando o construtor da classe pai
            self.cor = cor

        def fazer_som(self):
            return "Miau"

    gato = Gato("Mimi", "Branco")
    print(gato.nome)  # Mimi (herdado)
    print(gato.cor)   # Branco (definido na classe filha)
    print(gato.fazer_som())  # Miau

### Herança múltipla

    class Mamifero:
        def amamentar(self):
            return "Amamentando filhotes"

    class Cachorro(Animal, Mamifero):  # Herdando de Animal e Mamifero
        def fazer_som(self):
            return "Latido"

    dog = Cachorro("Rex")
    print(dog.amamentar())  # Amamentando filhotes (herdado de Mamifero)

> MRO (Method Resolution Order) em Python

O MRO (Method Resolution Order) é a ordem em que Python procura e executa métodos quando há `herança múltipla` ou `métodos sobrescritos`.

Python segue a regra C3 Linearization (ou algoritmo de MRO) para determinar essa ordem.

>📌 Como Python Define a Ordem de Busca?

1. Procura primeiro na própria classe.

2. Se não encontrar, procura na primeira classe pai listada na herança.

3. Se ainda não encontrar, segue a ordem da esquerda para a direita nas classes pai.

4. Segue subindo na hierarquia até object (a classe base de todas as classes).


## Polimorfismo

O polimorfismo é um princípio da Programação Orientada a Objetos (POO) que permite que ***diferentes classes tenham métodos com o mesmo nome, mas comportamentos diferentes***.

Isso facilita a reutilização de código e torna o código mais flexível.

    class Cachorro:
        def fazer_som(self):
            return "Au au"

    class Gato:
        def fazer_som(self):
            return "Miau"

> Polimorfismo + Herança

    class Animal:
        def __init__(self, nome):
            self.nome = nome

        def fazer_som(self):
            return "Som genérico"

(herdando a classe)

    class Cachorro(Animal):
        def fazer_som(self):
        return "Au au"

    class Gato(Animal):
        def fazer_som(self):
            return "Miau"

    animais = [Cachorro(), Gato(), Animal()]

    for animal in animais:
        print(animal.fazer_som()) 


O Python ***não suporta polimorfismo de sobrecarga*** como em `Java`e `C#` (métodos com mesmo nome, mas como parametros diferentes), mas é possível simular este comportamento.

#### Simulando sobrecarga

* usando argumento variáveis como *args e **kwargs

    ```
    class Calculadora:
    def somar(self, *args):
        return sum(args)

    calc = Calculadora()
    print(calc.somar(5))         # 5
    print(calc.somar(5, 10))      # 15
    print(calc.somar(1, 2, 3, 4)) # 10
    ```
* usando Funções Decoradoras

```
    def sobrecarga(funcao):
        def funcao_sobrecarga(self, *args, **kwargs):
            # fluxo de código
            return funcao(self, *args, **kwargs)
        
        return funcao_sobrecarga
    
    class Calculadora:
        @sobrecarga
        def funcao(self, parametros):
            # fluxo de código
```
* usando `functools.singledispatch` para Sobrecarga Baseada no Tipo

```
    from functools import singledispatch

    @singledispatch
    def processar(dado):
        raise NotImplementedError("Tipo não suportado")

    @processar.register(int)
    def _(dado):
        return f"Número inteiro: {dado}"

    @processar.register(str)
    def _(dado):
        return f"Texto: {dado.upper()}"

    @processar.register(list)
    def _(dado):
        return f"Lista com {len(dado)} elementos"

    print(processar(10))         # Número inteiro: 10
    print(processar("hello"))    # Texto: HELLO
    print(processar([1, 2, 3]))  # Lista com 3 elementos
```
se o singledispatch for usado em uma classe, o parametros `self` não considerada um parametro para ele. Deve-se usar ``@singledispatchmethod`

from functools import singledispatchmethod

    class Calculadora:
        @singledispatchmethod
        def somar(self, valor):
            raise NotImplementedError("Tipo não suportado")

        @somar.register(int)
        def _(self, valor):
            return valor + 10