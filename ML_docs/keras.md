# Keras

`valores padrão nas tabela de argumento estão em negrito`

## Funções

`image_dataset_from_directory`

Gerar um **tf.data.Dataset** a partir de arquivos em um diretório. Pode carregar/ler imagens `.jpg`, `.png`, `.bmp`, `.jpeg`, `.gif`.

## Argumentos

|Parâmetro|Descrição|Valores|
|---|---|---|
|**directory**|caminho do diretório onde contém as imagens|---|
|**labels**|os rótulos que classificam as imagens, se o valor for "*inferred*", ele irá levar em conta a divisão feita pelos sub-diretórios existentes no caminho acima, são definidos via os.walk(diretório) do Python|{"*inferred*", *None*}|
|**label_mode**|o modo como os rótulos serão representados no dataset carregdo|{"**int**", "*categorical*", "*binary*", *None*}|
|**batch_size**|tamanho dos batchs a serem usados no treino|---|
|**subset**|pode gerar subconjunto de treino, validação ou ambos (retorna uma tupla de datasets - treino e val), desde que o parâmetro *validation_split* esteja definido no **ImageDataGenerator**|{"*training*", "*validation*", "*both*"}|

