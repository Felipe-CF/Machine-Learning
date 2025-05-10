# Agrupamento - K-Means, Hierárquico, DBSCAN, Meanshift

O aprendizado por agrupamento é uma técnica que organiza um conjunto de dados em grupos ou `clusters`, onde os dados *dentro de cada grupo* são *mais semelhantes entre si* do que com os dados de outros grupos. É uma forma de aprendizado não supervisionado, pois *não requer rótulos* ou *categorias pré-definidas* para os dados.

## K-Means
Utiliza a `distância Euclidiana` para definir os grupos.

### Técnica básica
Após serem definidos os **klusters**, e através do `K-Means++`, são definidos os **centroides** que indicam a qual grupo pertencem os pontos. Depois eles *irão para o centro de seu grupo* e, caso algum **dado se aproxime mais de outro grupo** do que de seu centroide, ele é **remanejado**

### Definindo a quantidade de klusters/grupos - Elbow Method3
* Testa vários valores de k
* Define o número de clusters através do **ponto de inflexão** no gráfico do WCSS (Within Clusters Sum of Squares) em função do número de clusters.

![WCSS](https://imgur.com/g316Ili.jpg)


## Hierárquico
Aplicado em problemas de aprendizagem não supervisionada. Busca construir uma hierarquia de grupos, de forma aninhada, mesclando-os ou dividindo-os sucessivamente.

Esquematizada como **forma de uma árvore** e representada graficamente através de um dendrograma.

A **raiz da árvore** reúne **todas as amostras**, sendo as folhas com apenas uma amostra.

### 

|Tipos de Ligação|Descrição|
|-|-|
|**simples**|Conecta os clusters se baseando na menor das distâncias entre os pares de clusters|
|**média**|Conecta os clusters se baseando nas distâncias médias entre cada ponto em um cluster para cada particular em outro entre os pares de clusters|
|**completa**|Conecta os clusters se baseando na maior das distâncias entre os pares de clusters|
|**ward**|Conecta os clusters se baseando na soma das diferenças quadradas em todos os clsuters|


## DBSACN - Agrupamento Espacial de Aplicações com Ruído Baseado em Densidade

Mais rápido que o `kmeans` e o `hierárquico`, trabalhando bem com **outliers**. Bom para dados qeu contêm **clusters de densidade semelhante**, pois encontra amostras centrais de alta densidade e expande clsuters a partir delas. 

`Funcionamento`: começa em um ponto aleatório. Se sua vizinhaça *e* contiver pontos suficientes, *um cluster é iniciado*. Do contrário, ele é rotulado como **ruído**.

Requer 2 parâmetros:

```
 e (eps): raio de alance

min_smaples: quantidade mínima de pontos necessários para formar um cluster.
```

### Vantagens
* Não exige especificação do número de clusters, podendo encontrar eles de forma arbitrária.

* Robusto a outliers.

* Requer 2 parâmetros e é mais rápido.

### Desvantagens
* Pode não agrupar bem conjuntos de dados com grandes diferenças de densidades.

* Pode ser difícil escolher um limite de distância adequado.


## Meanshift

Ele cria uma **área de interesse**, onde um *vetor* indica em qual direção se encontra a *maior densidade de dados*. Após isso ele reconfigura o centro de interesse para mais próximo do **centro de massa**. Ele é aplicado em processamento de imagens e visão computacional. 

