# Configurando o early stopping

## Parâmetros aceitos

|**Hiperparametro**|**Descrição**|**Valor**|
|----|----|----|
|**monitor**|quantidade a ser avaliada|*default = "val_loss*, {"loss"}|
|**min_delta**|mudança mínima a ser considerada para "melhoria" (qualquer valor absoluto menor que ele não é considerado)| *default = 0*|
|**patience**|número de épocas sem melhoria para considerar a parada|*default = 0*|
|**mode**|define a maneira como a quantidade do parametro *monitor* deve ser considerada para avaliação|{"auto", "min", "max"}, *default = "auto": ja inferido a partir da quantidade monitorada, "min": treino para quando a quantidade para de diminuir*, "max": treino para quando a quantidade para de crescer* |
|**start_from_epoch**|numero de epocas para a espera começar|*default = 0*|
|**baseline**|é um valor absoluto acerca do *monitor* para que, caso o modelo não o alcance, ao atingir o valor de *patience*, o treino é interrompido|*default = None*|
|**restore_best_weights**|garante que você recebe a melhor versão do modelo, indepente se houve piora dele ao longo das épocas de treino |*default = False*|





