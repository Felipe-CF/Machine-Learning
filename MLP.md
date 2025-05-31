# Rede Neural Perceptron Multicamadas (MLP)

## Construtor da rede

A rede é iniciada com um list, onde cada elemento representará uma camada com n neurônios

    self.num_layers (quantidade de camadas)
    self.sizes (neuronios por camada)

**será tomada por exemplo [784, 30, 10]**

### Bias
Os bias (vieses) serão iniciados após a 1° camada, pois nesta apenas as entradas (dados) as alimentam.

    self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

Partiremos do indice 1 em diante, seguindo uma *distribuição normal* (média 0, desvio padrão 1), criando para cada camada **um vetor de y linhas e 1 coluna** com valores aleatórios.

O resultado será uma lista de *arrays NumPy*, onde cada array-i representa os vieses da camada-i. Desse modo, antes de aplicar a função de ativação o bias é adicionado:

![Exemplo](https://imgur.com/ogXGvRL.png)

Isso permite que o neurônio "desligue" ou "ligue" com mais facilidade, independentemente das entradas

**Exemplo**
* Para a camada oculta (30 neurônios): um array (30, 1) com 30 vieses.
* Para a camada de saída (10 neurônios): um array (10, 1) com 10 vieses.*

### Pesos
Iniciar cada peso entre as conexões dos neurônios. Haverá uma relação direta entra as camadas, porque cada neurônio da camada-k deverá se conectar com os da camada seguinte, até a camada de saída.

    self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

O loop vai percorrer a lista de camadas, pareando (x, y) elas até a saída, gerando as conexões com pesos. Cada par cria uma matriz de ***y linhas*** e ***x colunas*** com números aleatórios da distribuição normal.  

> y: Número de neurônios na camada de destino.

> x: Número de neurônios na camada de origem.

**Exemplo**

* Primeiro par: (784, 30) - Conexão da camada de entrada (784 neurônios) para a camada oculta (30 neurônios).

* Segundo par: (30, 10) - Conexão da camada oculta (30 neurônios) para a camada de saída (10 neurônios).


Cada elemento ***Wjk*** em uma matriz de pesos representa a força da conexão do ***k-ésimo neurônio na camada anterior*** para o ***j-ésimo neurônio na camada atual***.

**𝑤𝑗𝑘 : peso da conexão entrada k → oculto j**

### Resumo da Inicialização

A rede é "montada" com sua estrutura definida e seus parâmetros (pesos e vieses) são iniciados aleatoriamente. Essa inicialização aleatória é crucial para quebrar a simetria e permitir que os neurônios aprendam padrões diferentes.


## feedforward(self, a): Propagando a Entrada

Aqui recebemos um **vetor de ativação (a)**, que será a entrada da rede. Para simular a propagação pela rede nós *pareamos* os **biases(b)** e os **pesos entre camadas(w)**, faremos o **dot_product** entre (**w**, **a**).

    for b, w in zip(self.biases, self.weights):
        a = sigmoid(np.dot(w, a)+b)

Como **w** é um vetor (y, x) e **a** é (x, 1) teremos um vetor resultando (y, 1), e somando **b** da camada atual ao vetor resultante e aplicando a **função de ativação sigmoide**, vamos atualizar **a** para ser usado como entrada da próxima camada da rede. Assim teremos a propagação das entradas por toda a rede.



## Treinamento com Descida de Gradiente Estocástico

Aqui temos a principal função para treinamento da rede, pois implementa a descida do gradiente.

    sgd(self, training_data, epochs, mini_batch_size, eta, test_data=None)

`training_data` ⇒ lista de tuplas **(x, y)**, onde **x** é o **vetor de entrada** e **y** o **vetor de saída** esperado para **x**.

`epochs` ⇒ vezes em que a rede passará pelo conjunto de dados de treino.

`mini_batch_size` ⇒ quantidade de elementos nos subconjuntos para calculo do SGD.

`eta` ⇒ taxa de aprendizagem (*η*), hiperparâmetro crucial para definir o quanto os pesos serão ajustados a cada *época de treino*.

> Com um *η* muito grande, a rede pode pular o mínimo e não convergir. Se for muito pequeno, o treinamento será lento e pode ficar preso em soluções ruins.

Em cada época, os dado de treino são embaralhados antes de serem criados/separados os `mini_batchs` ou `mini_lotes`. Esse processo evita a repetição de dos lotes em cada época e consequentemente aprendizagem viciada.

### Mini Lotes
A criação desse lotes é feita através de slices da lista training_data. Faremos da variável 
`mini_batches` uma lista de "fatias" com  `len(mini_batches[i]) == mini_batch_size`.

    mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]

Desse modo o o treinamento será acelerado e ajuda a evitar mínimos locais:

🚀 1. Menos cálculos por passo (mais rápido por iteração)

>Treinar com todos os dados de uma vez (batch gradient descent) é muito pesado computacionalmente

🔁 2. Mais atualizações por época

>Em vez de esperar passar por todos os dados para atualizar os pesos, o modelo atualiza os pesos várias vezes por época — a cada mini-batch - aprendendo mais rápido.

🎲 3. Introduz aleatoriedade útil (melhor generalização)

> Como cada mini-batch contém dados diferentes, o gradiente calculado é uma estimativa ruidosa do gradiente real. Esse "ruído" ajuda a evitar mínimos locais ruins e melhora a capacidade de generalização da rede


### Resumo

O SGD é um processo iterativo. Em cada época, ele embaralha os dados, os divide em mini-lotes, e para cada mini-lote, ele ajusta os pesos e vieses da rede. Isso continua por várias épocas, permitindo que a rede "aprenda" os padrões nos dados e melhore sua capacidade de fazer previsões.

## Atualizando Pesos e Vieses
    update_mini_batch(self, mini_batch, eta)

É responsável por aplicar o algoritmo de `backpropagation` para calcular os gradientes e então atualizar os **pesos** e **vieses** da rede


### ∇ (nabla) 

Representa o gradiente de uma função - em relação aos *pesos* (nabla_w) e aos *biases* (nabla_b) e serão usados para acumular os gradientes.

    nabla_b = [np.zeros(b.shape) for b in self.biases]

### backprop nos batchs

Iremos então iterar sobre o `mini_batch` pareando entrada e saída e realizar o "*backprop*". Isso nos retornará os **gradientes para os biases e pesos** como listas.

    for x, y in mini_batch:
        delta_nabla_b, delta_nabla_w = self.backprop(x, y)

Para **cada par** desses serão calculados os **gradientes parciais** de biases e pesos — ou seja, **quanto cada peso/vies deveria mudar para aquele exemplo específico**.

Eles serão somados a cada iteração (entrada) para obtermos um **gradiente médio** daquele batch, suavizando as mudanças que viram na atualização dos parâmetros. 


### atualizando...
    self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
    
Esse passo é feito através da regra delta:

​![regra delta](https://imgur.com/DmT49Fp.png)
 
Por usamos o **gradiente médio** do batch, o `eta` é dividido pelo tamanho do mini-lote, fazendo com que a atualização reflita a média dos gradientes dos exemplos. Assim temos:

​![regra delta do gradiente médio](https://imgur.com/8MsBzgJ.png)





