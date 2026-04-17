# 📄 Revisão Crítica do Artigo – CrohNet (Formato IEEE Access)

## 🔎 Avaliação Geral

O artigo apresenta uma proposta relevante, focada na aplicação de CNNs compactas para detecção da Doença de Crohn, com ênfase em eficiência computacional (TinyML aplicado à saúde).

### ✔ Pontos fortes
- Relevância científica e social elevada
- Metodologia bem estruturada e reprodutível
- Foco em eficiência computacional (diferencial importante)

### ❌ Pontos críticos
- Validação experimental limitada
- Comparação insuficiente com o estado da arte
- Falhas na clareza científica em seções-chave

### 📊 Probabilidade de aceitação
- Atual: **60–70% (borderline)**
- Após melhorias: **80–90% (alto potencial)**

---

## 1) 📌 Título

### ✔ Avaliação
- Claro e direto
- Relacionado ao conteúdo

### ❌ Problemas
- Falta especificidade metodológica
- Não menciona:
  - Tipo de dado (endoscopia por cápsula)
  - Natureza da tarefa (classificação)

### 🔧 Sugestão
CrohNet: A Lightweight CNN for Crohn’s Disease Detection in Capsule Endoscopy Images under Low-Resource Constraints

---

## 2) 📄 Resumo

### ✔ Avaliação
- Contextualização adequada
- Problema bem definido

### ❌ Problemas críticos
- Ausência de métricas quantitativas
- Linguagem vaga ("alta taxa", "resultados consistentes")
- Metodologia superficial

### 📊 Estrutura atual

| Elemento    | Status |
|------------|--------|
| Introdução | ✔      |
| Problema   | ✔      |
| Metodologia| ⚠      |
| Resultados | ❌     |
| Conclusão  | ⚠      |

### 🔧 Melhorias necessárias
- Incluir dataset utilizado
- Apresentar métricas (AUC, accuracy, F1)
- Comparação com baseline
- Evidenciar redução de custo computacional

---

## 3) 📚 Introdução

### ✔ Pontos fortes
- Excelente contextualização
- Relevância bem explorada
- Problema claramente definido

### ❌ Problemas
- Gap científico não explícito
- Contribuições não listadas formalmente

### 🔧 Sugestão

Adicionar seção explícita:

This work makes the following contributions:
- Development of a lightweight CNN architecture
- Optimization for low-resource environments
- Evaluation using cross-validation
- Comparison with existing models

---

## 4) 🔗 Trabalhos Relacionados

### ✔ Pontos fortes
- Boa cobertura de modelos relevantes
- Uso do dataset CROHN-IPI

### ❌ Problemas críticos
- Falta análise comparativa estruturada
- Ausência de tabela comparativa
- Posicionamento da proposta pouco claro

### 🔧 Melhorias
- Criar tabela com:
  - Modelo
  - AUC
  - Parâmetros
  - Custo computacional
- Explicitar diferencial do CrohNet

---

## 5) ⚙️ Metodologia

### ✔ Pontos fortes
- Muito bem detalhada
- Reprodutível
- Pipeline completo
- Justificativas técnicas consistentes

### ⭐ Destaque
Seção forte e adequada para IEEE Access

### ⚠ Melhorias
- Reduzir excesso de detalhamento descritivo
- Melhorar organização visual
- Aprimorar diagrama da arquitetura

---

## 6) 📊 Resultados

### ✔ Pontos fortes
- Uso de validação cruzada
- Métricas completas
- Baixo desvio padrão (estabilidade)
- Comparação com CROHN-IPI

### ❌ Problemas críticos

#### 1. Análise superficial
- Falta interpretação dos resultados

#### 2. Comparação limitada
- Não inclui modelos leves (MobileNet, EfficientNet-lite)

#### 3. Falta análise estatística
- Ausência de testes de significância

#### 4. Falta contexto clínico
- Sensibilidade vs especificidade não discutidos

### 🔧 Melhorias
- Expandir análise dos achados
- Incluir comparação com modelos leves
- Discutir impacto clínico
- Analisar erros (FP/FN)

---

## 7) 🧾 Conclusão

### ✔ Pontos fortes
- Coerente com os resultados
- Retoma os objetivos

### ❌ Problemas
- Não discute limitações
- Pouco aprofundamento crítico

### 🔧 Melhorias
Adicionar:
- Limitações do estudo:
  - Dataset pequeno
  - Falta de validação externa
  - Risco de overfitting
- Trabalhos futuros mais concretos

---

## 🚨 Problemas Gerais Críticos

### ❗ 1. Idioma
- Artigo em português
- IEEE Access exige inglês
- Rejeição direta se não corrigido

### ❗ 2. Referências
- Não inseridas (apenas placeholders)
- Impacto direto na avaliação

### ❗ 3. Figuras
- Elementos incompletos ("a definir")
- Qualidade visual insuficiente

### ❗ 4. Erros técnicos
- Possível erro na fórmula de FPR:

Correto: FP / (FP + TN)

---

## ✅ Prioridades de Correção

### 🔥 Alta prioridade
1. Traduzir artigo para inglês
2. Reescrever resumo com métricas
3. Inserir referências corretamente

### ⚠ Média prioridade
4. Expandir análise de resultados
5. Melhorar comparação com estado da arte
6. Explicitar contribuições

### 🛠 Baixa prioridade
7. Ajustar figuras
8. Revisar equações e notação

---

## 💡 Conclusão Final

O artigo apresenta:

- ✔ Boa base técnica
- ✔ Relevância científica
- ✔ Aplicação prática importante

Porém, os principais riscos de rejeição são:

- ❌ Posicionamento científico insuficiente
- ❌ Validação experimental limitada
- ❌ Não conformidade com padrões IEEE