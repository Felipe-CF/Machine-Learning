# 5 - Separação Treino-Teste


## O que é?

Separar os dados que temos para treinamentos e testes.

## Código

    from sklearn.model_selection import train_test_split


    dados = pd.read_csv('C:\\Users\\FelipeCF\\Desktop\\Codigos\\Machine-Learning\\archive\\heart_tratado.csv',
                        sep=';', encoding='iso-8859-1')

    dados_tratados = pd.DataFrame.copy(dados)

    previsores = dados_tratados.iloc[:, 0:11].values

    alvo = dados_tratados.iloc[:, 11].values 

    previsores2 = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [1, 2, 6, 8, 10])], 
                                    remainder='passthrough').fit_transform(previsores)

    previsores2_escalonados = StandardScaler().fit_transform(previsores2)

    x_treino, x_teste, y_treino, y_teste = train_test_split(previsores2_escalonados, alvo, test_size=0.3, random_state=0)


