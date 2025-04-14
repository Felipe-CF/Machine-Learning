from bases.base_treino_teste import *
from sklearn.metrics import roc_auc_score
from bases.dados.dados_tratados_esc import *
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV


gbm = GradientBoostingClassifier(random_state=0)

parametro_grid = dict(
    n_estimators=[20, 50, 100, 250],
    learning_rate=[.05, .1, .5],
    max_depth=[1,2,3,4,5]
)

grid_search = GridSearchCV(gbm, parametro_grid, scoring='roc_auc', cv=4)

grid_search.fit(x_train, y_train)

print(grid_search.best_params_) # {'learning_rate': 0.05, 'max_depth': 2, 'n_estimators': 50}

resultado = pd.DataFrame(grid_search.cv_results_)

print(resultado.head(3))

# ordenando os melhores resultados
resultado.sort_values(by='mean_test_score', ascending=False, inplace=True)

resultado.reset_index(drop=True, inplace=True)

print(resultado[['param_max_depth', 'param_learning_rate', 'param_n_estimators',
                 'mean_test_score', 'std_test_score']].head())

