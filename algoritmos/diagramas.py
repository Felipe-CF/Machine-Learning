from dataset_mall import *
import plotly.express as px
import seaborn as sns

# histograma = px.histogram(dados_frame, x='Age', nbins=60)
# histograma.update_layout(width=600, height=400, title_text='Distribuição das idades')
# histograma.show()

print(dados_frame['Genre'].value_counts())

sns.countplot(x='Genre', data=dados_frame)