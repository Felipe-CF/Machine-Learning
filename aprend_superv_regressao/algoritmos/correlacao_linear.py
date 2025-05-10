from bases.dados_tratados import *
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.stats as stats

'''
RM: numero medio de comodos entre imoveis no bairro
LSTAT: porcentagem de proprietarios no bairro considerados classe baixa
PTRATIO: razao entre estudantes e professoras nas escolas
MEDV: valor medio das casas
'''


# plt.scatter(dados_tratados.RM, dados_tratados.MEDV)
# plt.title('Correlação')
# plt.xlabel('Quantidade de comodos')
# plt.ylabel('valor medio')
# plt.grid(False)


# plota todas as variaveis
# sns.pairplot(dados_tratados)


# analisar normalidade -gráfico Q-Q Plot
# se houver dispersão de pontos consideravel, não será viavel uma dist normal
# ==> fazer histograma, 
# stats.probplot(dados_tratados['MEDV'], dist='norm', plot=plt)
# plt.title('Normal Q-Q Plot')
# plt.show()


#   ===  analisar normalidade   === 

# Testes estatísticos 

'''
Hipóteses
H0 -> distrb. normal: p > 0.05
Ha -> distrb. não normal: p <= 0.05
'''

# Shapiro-Wilk (ate 5k de registros)
estatistica, p = stats.shapiro(dados_tratados.MEDV) 
print('Teste Shapiro-Wilk')
print('Estatistica do MEDV: {}'.format(estatistica))
print('P valor do MEDV: {}'.format(p))

# Lilliefors
# from statsmodels.stats.diagnostic import lilliefors

# estatistica, p = statsmodels.stats.diagnostic.lilliefors(dados_tratados.MEDV, dist='norm') 
# print('Teste Lilliefors')
# print('Estatistica do MEDV: {}'.format(estatistica))
# print('P valor do MEDV: {}'.format(p))


#   === Teste de correlação === 

'''
Hipóteses
H0 -> não  existe correlação > 0.05
Ha -> existe correlação <= 0.05

Pearson e Kendall são para distrib. normais, visto que desde os gráficos acima
é possível notar que não será normal.
'''

estatistica, p = stats.spearmanr(dados_tratados.MEDV, dados_tratados.RM) 
print('Teste Spearman')
print('Estatistica do MEDV: {}'.format(estatistica))
print('P valor do MEDV: {}'.format(p))

correlacoes = dados_tratados.corr(method='spearman')
print(correlacoes)

'''
melhor correlação: MEDV e LSTAT
'''
