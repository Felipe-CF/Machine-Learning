# Agrupamento - K-Means, Hierárquico

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


