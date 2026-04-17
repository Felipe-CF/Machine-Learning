# 📄 🔥 CHECKLIST RÁPIDO (POR SEÇÃO)

---

# 1) 📌 TÍTULO

## ❌ Problema
- Muito genérico (“CNN Compacta”)
- Não menciona **capsule endoscopy**
- Não deixa claro o **cenário de baixo recurso**

## ✅ Solução (copiar)
CrohNet: A Lightweight CNN for Crohn’s Disease Detection in Capsule Endoscopy Images under Low-Resource Constraints

---

# 2) 📄 RESUMO

## ❌ Problemas
- Sem números
- Sem dataset
- Sem comparação
- Linguagem vaga (“alta taxa”, “consistentes”)

## ✅ Solução (estrutura pronta)

Inclua:

- Dataset:
> using the CROHN-IPI dataset

- Métricas:
> achieving an AUC of 0.93 and accuracy of 0.85

- Eficiência:
> while reducing model size by approximately 47% compared to ResNet34

---

## ✍ Mini-template (pode adaptar)

This work proposes CrohNet, a lightweight convolutional neural network for automated classification of capsule endoscopy images. Experiments conducted on the CROHN-IPI dataset achieved an AUC of 0.93 and accuracy of 0.85. The proposed model reduces computational requirements by approximately 47% compared to ResNet34, enabling deployment in low-resource environments.

---

# 3) 📚 INTRODUÇÃO

## ❌ Problema
- Gap implícito, não explícito

## ✅ Solução (colar parágrafo)
However, most state-of-the-art approaches rely on deep architectures with high computational cost, limiting their applicability in low-resource environments such as public healthcare systems in developing regions.

---

## ❌ Problema
- Contribuições não listadas

## ✅ Solução (colar)
The main contributions of this work are:
- A lightweight CNN architecture for Crohn’s disease detection
- Optimization for low-resource hardware environments
- Evaluation using cross-validation on the CROHN-IPI dataset
- Analysis of computational efficiency and processing time

---

# 4) 🔗 RELATED WORK

## ❌ Problema
- Falta comparação estruturada

## ✅ Solução (fácil)

Criar tabela simples:

| Model     | AUC  | Parameters |
|----------|------|-----------|
| ResNet34 | 0.97 | 21M       |
| CrohNet  | 0.93 | 11M       |

---

## ❌ Problema
- Não menciona modelos leves existentes

## ✅ Solução (colar)
Although lightweight architectures such as MobileNet and EfficientNet have been proposed, their application to capsule endoscopy datasets for Crohn’s disease remains limited.

---

# 5) ⚙️ METODOLOGIA

## ❌ Problema
- Mistura implementação com metodologia

## ✅ Solução
Mover tudo que é:
- hardware
- tempo
- early stopping prático

👉 para subseção “Implementation”

---

## ❌ Problema
- Algumas decisões não justificadas

## ✅ Solução (exemplo pronto)
The choice of PReLU aims to mitigate the dying ReLU problem while maintaining computational efficiency.

---

# 6) 📊 RESULTADOS

---

## ❌ Problema 1: escolha do melhor fold

### ✅ Solução
Trocar:

❌
modelo final = fold 4

✔
The model performance is reported as the average across all folds to ensure robustness.

---

## ❌ Problema 2: custo computacional pouco explorado

### ✅ Solução (colar)
CrohNet reduces the number of parameters by approximately 47% compared to ResNet34, while maintaining competitive performance, highlighting its suitability for low-resource environments.

---

## ❌ Problema 3: não explorou limitação de hardware

### ✅ Solução (forte)
Training deeper architectures such as ResNet34 under the same hardware constraints would require approximately 30 days, making them impractical in low-resource settings.

---

## ❌ Problema 4: falta interpretação clínica

### ✅ Solução (colar)
From a clinical perspective, sensitivity is particularly important, as false negatives may delay diagnosis and treatment.

---

# 7) 🧾 CONCLUSÃO

---

## ❌ Problema: não fala limitações

### ✅ Solução (colar)
This study has some limitations, including the use of a single dataset and the lack of external validation.

---

## ❌ Problema: não explica limitação de dados

### ✅ Solução
The limited availability of publicly accessible capsule endoscopy datasets remains a challenge for further validation.

---

# 🚨 ERRO CRÍTICO (OBRIGATÓRIO CORRIGIR)

## ❌ Errado
TFP = FP / (FP + FN)

## ✅ Correto
FPR = FP / (FP + TN)

---

# ⚡ MELHORIAS RÁPIDAS (ALTO IMPACTO / BAIXO CUSTO)

- Adicionar números no resumo
- Explicitar gap
- Listar contribuições
- Não escolher “melhor fold”
- Interpretar tabelas (não só mostrar)
- Adicionar limitações
- Corrigir fórmula

---

# 🧠 RESUMO FINAL

Você NÃO precisa:
- treinar novos modelos
- mudar arquitetura
- usar GPU melhor

Você só precisa:
👉 explicar melhor o que já fez