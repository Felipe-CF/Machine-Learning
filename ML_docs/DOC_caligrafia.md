

**Bias (viés)**: São vetores coluna inicializados para cada camada, exceto a camada de entrada.

- Para uma rede [2, 3, 1], temos bias para a camada oculta (3×1) e para a camada de saída (1×1).
- Matematicamente, o bias permite que a função de ativação seja deslocada, aumentando a flexibilidade do modelo.



**Weights (pesos)**: São matrizes que conectam neurônios entre camadas adjacentes.

- A matriz de pesos entre a camada de entrada e a camada oculta tem dimensão (3×2).
- A matriz de pesos entre a camada oculta e a camada de saída tem dimensão (1×3).
- Cada elemento w_jk representa o peso da conexão do neurônio k na camada atual para o neurônio j na próxima camada.



**Distribuição Normal**: Os pesos e bias são inicializados com uma distribuição normal (gaussiana) com média 0 e desvio padrão 1.

- Esta inicialização aleatória é crucial para quebrar a simetria entre os neurônios.
- Se todos os pesos fossem inicializados com o mesmo valor, todos os neurônios aprenderiam a mesma função.