# Decoradores

## 📌 O Que é um Decorator em Python?

Um decorator é uma **função que recebe outra função como argumento**, adiciona algum comportamento extra a ela e retorna uma nova função. Isso permite modificar o comportamento de funções e métodos sem alterar seu código original.

Exemplo

    def meu_decorator(func):
        def wrapper():
            print("Iniciando função...")
            func()  # Chama a função original
            print("Finalizando função...")
        return wrapper  # Retorna a nova função

    @meu_decorator
    def saudacao():
        print("Olá, mundo!")

    saudacao()

> os mesmos argumentos do `wrapper` devem ser passados ao chamar a função original

    def meu_decorator(func):
        def wrapper(*args, **kwargs):  # Captura os argumentos
            print(f"Chamando {func.__name__} com {args}, {kwargs}")
            return func(*args, **kwargs)  # Repassa os argumentos
        return wrapper

    @meu_decorator
    def soma(a, b):
        return a + b

    print(soma(2, 3))  # ✅ Funciona corretamente!


## 🚀 Resumo
✔ Decorators são funções que modificam outras funções.

✔ Podem ser usados para adicionar logs, validações, medições de tempo, etc.

✔ Podem aceitar argumentos para personalizar o comportamento.

✔ Podem ser usados em funções normais ou métodos de classe.