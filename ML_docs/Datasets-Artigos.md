# Vamos salvar a resposta anterior em formato Markdown (.md) com todos os dados incluídos.
markdown_content = """
# Datasets Públicos de Imagens para Doenças Inflamatórias Intestinais (IBD)

Este documento reúne datasets públicos de imagens rotuladas referentes a doenças inflamatórias intestinais (Inflammatory Bowel Disease - IBD), úteis para a construção de classificadores baseados em Redes Neurais Convolucionais (CNN). Todos os datasets mencionados estão vinculados a publicações científicas.

---

## 1. Pediatric IBD (CHOC) – Whole Slide Histologia (H&E)
- **Descrição**: 951 varreduras de seções de tecido H&E de 18 pacientes (pediátricos), com pixel-level labels de normal vs. anormal, além de anotações por lâmina, seção e paciente.
- **Tamanho**: 951 imagens.
- **Origem / Artigo**: Martin‑King et al., Cell Image Library (2024).
- **Link**: https://cellimagelibrary.org/groups/55420

---

## 2. IBDColEpi – Biópsias de cólon HE e CD3
- **Descrição**: 140 HE-stained e 111 CD3-stained cortes de biópsias com epitélio anotado (patches + WSI + segmentações).
- **Tamanho**: centenas de imagens WSI e milhares de patches.
- **Origem / Artigo**: André Pedersen et al., NTNU/St. Olavs Hospital.
- **Link**: https://huggingface.co/datasets/andreped/IBDColEpi

---

## 3. Colonoscopy Endoscopic Images – Crohn vs. UC vs. Normais
- **Descrição**: 47.322 imagens de 1.576 pacientes (2018–2020), rotuladas como normais, UC ou CD.
- **Tamanho**: dezenas de milhares de imagens.
- **Origem / Artigo**: Frontiers in Medicine (2022).
- **Link do artigo**: https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2022.854677/full


---

## 4. Hyper-Kvasir – Gastrointestinal geral com UC
- **Descrição**: 110.079 imagens e 374 vídeos; 10.662 imagens rotuladas, incluindo 851 de colite ulcerativa (UC) com escala MES.
- **Tamanho**: grande.
- **Origem / Artigo**: Borgli et al., Hyper-Kvasir dataset.
- **Link**: https://datasets.simula.no/hyper-kvasir/

---

## 5. ERS Dataset – Rotulagens detalhadas MST 3.0
- **Descrição**: ~6.000 imagens rotuladas com precisão e ~115.000 rotulagens aproximadas; 27 tipos de achados.
- **Tamanho**: ~121.000 imagens.
- **Origem / Artigo**: Cychnerski et al. 2022.
- **Link**: https://data.mendeley.com/datasets/g2r6xg7cgh

---

## 6. CrohnIPI – Capsule Endoscopy para Crohn
- **Descrição**: 3.498 imagens de vídeo-cápsula endoscopia, incluindo 6 tipos de lesões inflamatórias (úlcera, estenose, edema, etc).
- **Tamanho**: milhares de imagens.
- **Origem / Artigo**: Dataset CrohnIPI.
- **Link**: https://figshare.com/articles/dataset/CrohnIPI/20490318

---

## 7. Link LIMUC

https://zenodo.org/records/5827695


## Comparativo Rápido

| Dataset                     | Modalidade         | Crohn vs UC      | Quantidade         | Tipo de dados                        |
|----------------------------|--------------------|------------------|--------------------|--------------------------------------|
| Pediatric IBD (CHOC)       | Histopatologia     | IBD (anormal)     | ~951 WSIs          | H&E, labels por seção e patologia    |
| IBDColEpi                  | Biópsias HE/CD3    | Ativa vs inativa  | ~250 biopsias      | WSI + patches + segmentações         |
| CNN endoscopy (Frontiers)  | Colonoscopia       | CD / UC / normal  | ~47.300 imagens    | Imagens HD com rótulos clínicos      |
| Hyper‑Kvasir               | GI endoscopia      | UC (851 imagens)  | ~110.000 imagens   | Imagens + vídeos, multiclasses       |
| ERS                        | Endoscopia + cápsula| Inflamações, UC  | ~121.000 imagens   | Anotações MST 3.0 + masks             |
| CrohnIPI                   | Capsule Endoscopy  | Crohn lesions     | ~3.500 imagens     | Lesões inflamatórias detalhadas      |

---

Gerado por assistente AI para fins acadêmicos.

"""

# Salvando como arquivo .md
file_path = "/mnt/data/datasets_IBD_CNN_revisao.md"
with open(file_path, "w", encoding="utf-8") as f:
    f.write(markdown_content)

file_path
