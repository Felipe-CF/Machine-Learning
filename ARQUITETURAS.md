## Justificativa para Seleção de Arquiteturas

Embora existam arquiteturas modernas amplamente utilizadas na literatura, como MobileNet e EfficientNet, sua adoção não foi considerada neste estudo devido a fatores relacionados ao escopo e às limitações computacionais disponíveis.

Primeiramente, arquiteturas como MobileNet foram projetadas com foco em aplicações embarcadas e dispositivos móveis, priorizando eficiência em inferência em ambientes com restrições de hardware específicas. No entanto, este trabalho não tem como objetivo a implantação em dispositivos móveis, mas sim a análise de desempenho em ambiente controlado, o que reduz a relevância da adoção dessa classe de modelos.

Adicionalmente, modelos como EfficientNet, apesar de apresentarem excelente desempenho e eficiência, possuem um custo computacional significativo durante o treinamento. Estudos prévios indicam que o treinamento dessas arquiteturas frequentemente demanda infraestrutura de alto desempenho, o que não está alinhado com as condições deste trabalho.

Dessa forma, optou-se por utilizar arquiteturas que ofereçam um equilíbrio entre desempenho e viabilidade computacional, permitindo a execução completa dos experimentos em ambiente acadêmico com recursos limitados, garantindo também a reprodutibilidade dos resultados.

Por fim, destaca-se que a inclusão de arquiteturas mais complexas e computacionalmente custosas é considerada uma direção promissora para trabalhos futuros, especialmente em cenários com maior disponibilidade de recursos computacionais.


## Justificativa dos Modelos

A escolha das arquiteturas utilizadas neste estudo foi guiada por restrições computacionais e pelo escopo do trabalho. Modelos como MobileNet não foram considerados, pois o foco deste estudo não envolve aplicações em dispositivos móveis.

Além disso, arquiteturas como EfficientNet, embora eficientes, apresentam elevado custo computacional durante o treinamento, tornando sua utilização inviável no ambiente disponível.

Assim, foram selecionados modelos compatíveis com recursos limitados, garantindo a execução completa dos experimentos e a reprodutibilidade dos resultados. A avaliação de arquiteturas mais complexas é deixada como trabalho futuro.