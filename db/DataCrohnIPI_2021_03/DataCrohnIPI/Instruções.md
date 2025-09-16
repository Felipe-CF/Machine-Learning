CROHN-IPI Dataset
=======

https://crohnipi.ls2n.fr/

O dataset CrohnIPI contém 3491 imagens de cápuslas endoscópicas. Cada imagem é associada comum dos seguintes rótulos não exludentes:

- E : Eridema
- O : Edema
- AU : Afitoide ulcerativa
- U3-10 : Ulceração entre 3mm e 10mm
- U>10 : Ulceração acima de 10mm
- S : Estenose
- N : Normal

Esse dataset é constituído de um direótrio contendo todas as imagens e um arquivo .csv com 3 colunas:
- A primeira para o nome do frame.
- Segunda para o nome do rótulo
- Como esse dataset foi desenvolvido primariamente para comparação entre modelos de classificação, nós incentivamos a validação cruzada K-fold, então a terceira coluna propõe o número do conjunto de teste (sendo 80% do de cada conjunto para treinamento e 20% para validação com 54 validações cruzadas)
