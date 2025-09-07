import matplotlib.pyplot as plt

# Dados extraídos manualmente dos logs de treinamento
# auc (treinamento) e val_auc (validação)
train_auc = [0.7046, 0.7972, 0.8151, 0.9346, 0.8369, 0.8789, 0.8482, 0.7712, 0.8566, 0.8392, 0.8426, 0.8340, 0.8346, 0.7126, 0.8509, 0.7816, 0.8456, 0.7428, 0.8527, 0.7956, 0.8715, 0.8128, 0.8731, 0.8646, 0.8741, 0.8740, 0.8638, 0.7728, 0.8301, 0.8285, 0.8558, 0.9860, 0.8581, 0.8324]
val_auc = [0.7798, 0.7728, 0.8143, 0.8150, 0.8276, 0.8240, 0.8247, 0.8262, 0.8285, 0.8352, 0.8313, 0.8199, 0.7235, 0.7405, 0.8484, 0.8478, 0.7909, 0.7919, 0.8104, 0.8086, 0.8645, 0.8659, 0.6528, 0.4844, 0.4753, 0.4749, 0.8243, 0.8197, 0.8267, 0.8261, 0.8442, 0.8515, 0.8250, 0.7747]

# binary_crossentropy (treinamento) e val_binary_crossentropy (validação)
train_bce = [0.7811, 0.3562, 0.3236, 0.1913, 0.2976, 0.2690, 0.2942, 0.3473, 0.2850, 0.2993, 0.2917, 0.3137, 0.2972, 0.3863, 0.2820, 0.3395, 0.2879, 0.3719, 0.2794, 0.3517, 0.2672, 0.3232, 0.2691, 0.2687, 0.2681, 0.2752, 0.2749, 0.3372, 0.2947, 0.3100, 0.2796, 0.1702, 0.2800, 0.2911]
val_bce = [0.6851, 0.7022, 0.4268, 0.4241, 0.4745, 0.4724, 0.5081, 0.5081, 0.3002, 0.2992, 0.3663, 0.3777, 0.3810, 0.3764, 0.2826, 0.2830, 0.3697, 0.3579, 0.2911, 0.2947, 0.2960, 0.2914, 0.4046, 0.5404, 2.6584, 2.8862, 0.2945, 0.2973, 0.2828, 0.2826, 0.3245, 0.3127, 0.3434, 0.3609]

# loss (treinamento) e val_loss (validação)
train_loss = [28277.7051, 1130.6082, 2612.4626, 257.9387, 131.5157, 221.3773, 98.0668, 32.3286, 24.8172, 0.7888, 2.1981, 0.8350, 0.5622, 0.7274, 0.5036, 0.4729, 0.4166, 0.4742, 0.3792, 0.4970, 0.4240, 0.4834, 0.4236, 0.4597, 0.4467, 0.4467, 0.4566, 0.4800, 0.4429, 0.4313, 0.4159, 0.3464, 0.4435, 0.4060]
val_loss = [1130.9370, 1194.8818, 258.1741, 241.4334, 221.5829, 217.4912, 32.4895, 32.6395, 0.7897, 0.8288, 0.8876, 0.8794, 0.7221, 0.7142, 0.4160, 0.4165, 0.4721, 0.4600, 0.4364, 0.4404, 0.4563, 0.4512, 0.5956, 0.7312, 2.8299, 3.0580, 0.4372, 0.4398, 0.4041, 0.4036, 0.5008, 0.4888, 0.4584, 0.4761]


epochs = range(1, len(train_auc) + 1)

# Define o estilo para ser mais parecido com o seu exemplo
plt.style.use('seaborn-v0_8-whitegrid') # Ou 'ggplot', ou 'fivethirtyeight' se preferir

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6)) # 3 subplots agora
fig.suptitle('Histórico de Treinamento e Validação', fontsize=16, y=1.02) # Título centralizado

# --- Gráfico 1: AUC ---
ax1.plot(epochs, train_auc, 'b-', label='Treinamento AUC')
ax1.plot(epochs, val_auc, 'r--', label='Validação AUC') # Linha tracejada vermelha
ax1.set_title('AUC vs. Épocas')
ax1.set_xlabel('Épocas')
ax1.set_ylabel('AUC')
ax1.set_ylim([0.75, 1.01]) # Ajuste o limite Y para focar na região de interesse
ax1.legend()
ax1.grid(True)

# --- Gráfico 2: Binary Cross-Entropy ---
ax2.plot(epochs, train_bce, 'b-', label='Treinamento BCE')
ax2.plot(epochs, val_bce, 'r--', label='Validação BCE') # Linha tracejada vermelha
ax2.set_title('Binary Cross-Entropy vs. Épocas')
ax2.set_xlabel('Épocas')
ax2.set_ylabel('Binary Cross-Entropy')
ax2.set_ylim([0.05, 0.55]) # Ajuste o limite Y conforme o exemplo
ax2.legend()
ax2.grid(True)

# --- Gráfico 3: Loss ---
ax3.plot(epochs, train_loss, 'b-', label='Treinamento Loss')
ax3.plot(epochs, val_loss, 'r--', label='Validação Loss') # Linha tracejada vermelha
ax3.set_title('Loss vs. Épocas')
ax3.set_xlabel('Épocas')
ax3.set_ylabel('Loss')
ax3.set_ylim([0.05, 0.55]) # Ajuste o limite Y conforme o exemplo
ax3.legend()
ax3.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96]) # Ajusta o layout para evitar sobreposição
plt.show()