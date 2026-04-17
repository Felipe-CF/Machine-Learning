# 📄 Revisão Crítica do Manuscrito – CrohNet (IEEE Access)

Esta é uma análise criteriosa do manuscrito *"CrohNet: Uma CNN Compacta para Detecção da Doença de Crohn com Requisitos Computacionais Reduzidos"* para a revista *IEEE Access*.

O trabalho apresenta uma proposta relevante de aplicação de Tiny Machine Learning (TinyML) em contextos de saúde pública, especificamente para o diagnóstico via cápsula endoscópica em regiões com limitações de hardware.

---

## 1) 📌 Título

### ✔ Análise
- O título é claro, específico e reflete diretamente:
  - A contribuição técnica (compactação/eficiência)
  - A aplicação clínica (Doença de Crohn)

### 🔧 Sugestão
Como a revista IEEE Access é internacional, o título deve ser traduzido para o inglês antes da submissão final:

CrohNet: A Compact CNN for Crohn's Disease Detection with Reduced Computational Requirements

---

## 2) 📄 Resumo (Abstract)

### ✔ Mapeamento
O resumo contempla:

- Introdução: aumento da incidência e uso de cápsulas
- Metodologia: redes neurais para pré-classificação
- Resultados: alta taxa de acerto em hardware limitado
- Conclusão: potencial de aplicação prática

### ❌ Ponto Crítico
- Faltam dados quantitativos específicos:
  - AUC
  - Acurácia
- Isso reduz o impacto científico do resumo

---

## 3) 📚 Introdução

### ✔ Contexto e Relevância
- Muito bem fundamentados
- Destaque para:
  - Impacto socioeconômico no Nordeste brasileiro
  - Vantagem da cápsula sobre a colonoscopia

### ✔ Problema / Gap
- Identifica corretamente que:
  - Modelos robustos exigem hardware de alto custo
  - Isso limita o uso em sistemas públicos

### ✔ Contribuições
- CrohNet como arquitetura leve
- Uso de PReLU adaptativa
- Técnica de ponderação de classes

---

## 4) 🔗 Trabalhos Relacionados

### ✔ Análise
- Bom posicionamento em relação ao estado da arte:
  - Trabalhos de Wang et al.
  - Trabalhos de Polat et al.
- Uso do projeto CROHN-IPI como baseline

### ⚠ Sugestão
- Explicar melhor por que arquiteturas leves como:
  - MobileNet
  - SqueezeNet
não foram utilizadas

---

## 5) ⚙️ Metodologia

### ✔ Clareza
- Seção bem detalhada
- Inclui:
  - Dataset: 3498 imagens
  - Validação cruzada (k-fold com 5 folds)
  - Classificação binária

### ✔ Componentes
- Data Augmentation
- Normalização
- Uso de PReLU (evita "Dying ReLU")

### ✔ Hardware
- Detalhamento importante:
  - Intel Core i5
  - GPU integrada

→ Reforça a proposta de baixo custo computacional

---

## 6) 📊 Resultados

### ✔ Análise
- AUC: **93,7%**
- Comparação de tamanho:
  - CrohNet: 42,69 MB
  - ResNet34: 81,30 MB

### ✔ Ponto Positivo
- Simulação prática:
  - 50.000 imagens processadas
  - Tempo: ~1,8 horas (6800s)

→ Demonstra viabilidade real

### ⚠ Ponto de Atenção
- Desempenho inferior ao CROHN-IPI em:
  - Acurácia: 86,5%
  - Sensibilidade: 86,5%

- CROHN-IPI:
  - Acurácia: 93,7%
  - Sensibilidade: 92,4%

### ❗ Implicação crítica
- Maior risco de falsos negativos
- Isso deve ser discutido sob a perspectiva clínica

---

## 7) 🧾 Conclusão e Trabalhos Futuros

### ✔ Fechamento
- Resume corretamente:
  - Desempenho competitivo
  - Redução significativa de armazenamento

### ✔ Perspectivas
- Integração em aplicações offline
- Uso em comunidades rurais

→ Forte impacto social

---

## 📊 Avaliação de Chances de Aprovação

### 📈 Chances
**Médias-Altas**, desde que ajustes sejam realizados

- Tema alinhado com tendências:
  - TinyML
  - Green AI

---

## 🚨 Pontos Críticos para Melhoria

### ❗ 1. Língua do Manuscrito
- IEEE Access exige **Inglês**
- Tradução técnica obrigatória

### ❗ 2. Referências Bibliográficas
- Atualmente ausentes ([?])
- Necessário:
  - Embasamento robusto
  - Trabalhos recentes (últimos 3–5 anos)

### ❗ 3. Qualidade das Figuras
- Figuras devem estar:
  - Em alta resolução (300 DPI ou vetorial)
  - Com legendas em inglês

- Problema identificado:
  - Figura marcada como "a definir"

### ❗ 4. Discussão de Trade-off
- Redução de sensibilidade (~6%) precisa ser melhor discutida

#### Pontos a abordar:
- Impacto de falsos negativos
- Justificativa clínica
- Relação custo-benefício:
  - Acessibilidade vs desempenho

---

## 💡 Conclusão Final

O trabalho apresenta:

- ✔ Base experimental sólida
- ✔ Clareza de propósito
- ✔ Forte impacto social

### ⚠ Riscos de rejeição
- Falta de referências
- Texto fora do padrão (idioma)
- Discussão insuficiente do trade-off clínico

---

Deseja que eu ajude a estruturar a discussão sobre a redução da sensibilidade para mitigar críticas dos revisores?