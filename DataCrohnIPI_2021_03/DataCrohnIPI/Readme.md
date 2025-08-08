CROHN-IPI Dataset
=======

https://crohnipi.ls2n.fr/

The CrohnIPI dataset contains 3484 images from endoscopic video capsules. Each image is associated with one of the following 7 labels:

- E : Erythème
- O : Edema
- AU : Aphthoid ulceration
- U3-10 : Ulcération between 3mm and 10mm
- U>10 : Ulceration over 10mm
- S : Stenosis
- N : Normal

This dataset is constituted of one folder containing all images and one csv file that contains 3 columns :
- The first one for the frame name.
- The second one for the label
- As this dataset is mainly design for comparison between classification models, we encourage K-fold cross-validation, so the third column propose the test set number. (with 80% of the dataset for each training and 20% for each testing and so 5 time cross-validations)

Theses images are for research use only, under the creative commons license [CC BY-NC-ND](https://creativecommons.org/licenses/by-nc-nd/2.0/)

Copyright: University of Nantes, CHU de Nantes, LS2N, IMAD

Contact: harold.mouchere@univ-nantes.fr

If you use this data in a paper, please cite these references:

  Added value of a multi-expert annotation of Crohn’s disease images of the small bowel for automatic detection of lesions using a convolutional recurrent attention neural network
  A de Maissin, R Vallée, M Flamant, M Fondain Bossiere, C Le Berre, A Coutrot, N Normand, H Mouchère, S Coudol, C Trang, A Bourreille
  Endoscopy International Open
  (Accepted)

  CrohnIPI: An endoscopic image database for the evaluation of automatic Crohn's disease lesions recognition algorithms.
  Rémi Vallée, A de Maissin, A. Coutrot, H. Mouchère, A. Bourreille, N. Normand
  Proc. SPIE 11317, Medical Imaging 2020: Biomedical Applications in Molecular, Structural, and Functional Imaging, 113171Q (28 February 2020)
  https://doi.org/10.1117/12.2543584
