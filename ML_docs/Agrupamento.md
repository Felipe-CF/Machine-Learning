# Agrupamento - K-Means

O aprendizado por agrupamento é uma técnica que organiza um conjunto de dados em grupos ou `clusters`, onde os dados *dentro de cada grupo* são *mais semelhantes entre si* do que com os dados de outros grupos. É uma forma de aprendizado não supervisionado, pois *não requer rótulos* ou *categorias pré-definidas* para os dados.

## K-Means
Utiliza a `distância Euclidiana` para definir os grupos.

### Técnica básica
Após serem definidos os **klusters**, e através do `K-Means++`, são definidos os **centroides** que indicam a qual grupo pertencem os pontos. Depois eles *irão para o centro de seu grupo* e, caso algum **dado se aproxime mais de outro grupo** do que de seu centroide, ele é **remanejado**

### Definindo a quantidade de klusters/grupos - Elbow Method3
* Testa vários valores de k
* Define o número de clusters através do **ponto de inflexão** no gráfico do WCSS (Within Clusters Sum of Squares) em função do número de clusters.

![WCSS](https://imgur.com/g316Ili.jpg)


