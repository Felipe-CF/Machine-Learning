# Localização dos Datasets Utilizados

Nesse arquivo estará os links referente ao download de todos os datasets de grande volume utilizados pelo Deep Learning Book, como para qualquer outro projeto/arquivo existente neste repositório.

## ConvNet's - dogs and cats dataset

[**Baixar Dataset via Microsoft (787 MB)**](https://www.microsoft.com/en-us/download/details.aspx?id=54765)


## 📁 CrohnIPI – Capsule Endoscopy para Crohn

>🧬 Origem e contexto
Coletado por pesquisadores da Universidade de Medicina de Varsóvia, Polônia, o dataset CrohnIPI foi desenvolvido para auxiliar no diagnóstico automatizado da Doença de Crohn por meio de imagens de vídeo-cápsula endoscópica.

> Link
https://figshare.com/articles/dataset/CrohnIPI/20490318


> 📊 Descrição e rotulações
O dataset contém 3.498 imagens de vídeo-cápsula endoscópica, incluindo seis tipos de lesões inflamatórias:

`Úlcera`

`Estenose`

`Edema`

`Erosão`

`Fístula`

`Hemorragia`

As imagens foram rotuladas manualmente por especialistas para facilitar o treinamento de modelos de aprendizado de máquina.

> 💾 Estrutura e tamanho dos arquivos

3.498 imagens de vídeo-cápsula endoscópica.

Tamanho total não especificado no repositório.

Formatos de imagem padrão para análise em aprendizado de máquina.

> 📏 Usos e boas práticas
Ideal para tarefas de classificação e detecção de lesões associadas à Doença de Crohn em imagens de vídeo-cápsula. Recomenda-se dividir os dados respeitando a independência entre pacientes para evitar viés, e utilizar as anotações fornecidas para treinar e validar modelos de inteligência artificial.

## 📁 LIMUC – Labeled Images for Ulcerative Colitis 

> 🧬 Origem e contexto
Coletado no Departamento de Gastroenterologia da Marmara University School of Medicine, Turquia. Imagens obtidas de 1043 procedimentos de colonoscopia realizados entre dezembro de 2011 e julho de 2019 em 564 pacientes com colite ulcerativa.

> Link

https://zenodo.org/records/5827695

> 📊 Descrição e rotulações
Total de 11.276 imagens endoscópicas, com classificação por Mayo Endoscopic Score (MES):

    Mayo 0: 6.105 imagens (54,14%)

    Mayo 1: 3.052 (27,70%)

    Mayo 2: 1.254 (11,12%)

    Mayo 3: 865 (7,67%) 

Etiquetagem feita por dois gastroenterologistas; caso de discordância, um terceiro avaliador definiu o score final por maioria 
Zenodo
.

> 💾 Estrutura e tamanho dos arquivos
A estrutura fornecida inclui:

    patient_based_classified_images.zip (~1,9 GB)
    train_and_validation_sets.zip (~1,6 GB)
    test_set.zip (~287 MB) 

Total aproximado: ~3,8 GB compactados.

As imagens são .bmp de resolução 352 × 288 pixels, conforme informações técnicas nos artigos.

> 📏 Usos e boas práticas
Os autores sugerem split por nível de paciente (não por imagem) para evitar viés por sobreposição de imagens similares em subsets separados. Scripts para criar divisões em 10‑fold cross‑validation estão disponíveis no repositório GitHub vinculado ao dataset


## 📁 HyperKvasir – Comprehensive Multi-class Gastrointestinal Image and Video Dataset

> 🧬 Origem e contexto
Coletado no Hospital Bærum, Noruega, durante exames reais de endoscopia gastrointestinal, incluindo gastroscopia e colonoscopia. O conjunto visa apoiar o desenvolvimento de sistemas automáticos para diagnóstico assistido por computador em gastroenterologia.

> Link
https://datasets.simula.no/hyper-kvasir/

> 📊 Descrição e rotulações
O dataset contém aproximadamente 110.079 imagens e 374 vídeos classificados em múltiplas categorias que incluem marcos anatômicos, achados patológicos e imagens normais. As imagens e vídeos foram rotulados por endoscopistas experientes, abrangendo diversas classes clínicas e anatômicas para diagnóstico assistido.

> 💾 Estrutura e tamanho dos arquivos

110.079 imagens com resolução variável.

374 vídeos contendo mais de 1 milhão de frames no total.

Dados disponíveis em formatos comuns para facilitar uso em aprendizado de máquina.

> 📏 Usos e boas práticas
Ideal para tarefas de classificação, segmentação, detecção e análise de imagens e vídeos endoscópicos. Recomenda-se dividir os dados respeitando a independência entre pacientes para evitar viés, e usar as anotações fornecidas para treinar e validar modelos de inteligência artificial.