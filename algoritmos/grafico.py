import matplotlib.pyplot as plt
import numpy as np

# Configurações
input_nodes = 30
hidden_nodes = 20
output_nodes = 1

# Posições dos nós
input_x = np.zeros(input_nodes)
input_y = np.linspace(0, 1, input_nodes)

hidden_x = np.ones(hidden_nodes) * 0.5
hidden_y = np.linspace(0.1, 0.9, hidden_nodes)

output_x = np.ones(output_nodes)
output_y = np.ones(output_nodes) * 0.5

# Criar figura
plt.figure(figsize=(10, 8))

# Desenhar conexões entre camada de entrada e oculta
for i in range(input_nodes):
    for h in range(hidden_nodes):
        plt.plot([input_x[i], hidden_x[h]], [input_y[i], hidden_y[h]], 'gray', alpha=0.1)

# Desenhar conexões entre camada oculta e saída
for h in range(hidden_nodes):
    plt.plot([hidden_x[h], output_x[0]], [hidden_y[h], output_y[0]], 'gray', alpha=0.3)

# Desenhar nós
plt.scatter(input_x, input_y, s=100, c='blue', label='Camada de Entrada (30 nós)')
plt.scatter(hidden_x, hidden_y, s=100, c='green', label='Camada Oculta (20 nós)')
plt.scatter(output_x, output_y, s=100, c='red', label='Camada de Saída (1 nó)')

# Adicionar legendas e título
plt.title('Multilayer Perceptron (30-20-1)')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
plt.axis('off')

# Salvar como PNG
plt.savefig('multilayer_perceptron.png', dpi=300, bbox_inches='tight')
plt.show()