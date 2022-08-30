get_ipython().system('pip install pandas')
get_ipython().system('pip install sklearn')
get_ipython().system('pip install lightgbm')
get_ipython().system('pip install xgboost')

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

def model_rank(data, methods, parameters=None):
    model_name =  {"RF" : RandomForestClassifier(),
                "LGBM" : LGBMClassifier(),
                "XGB" : XGBClassifier(),
                "EXtra" : ExtraTreesClassifier(),
                "Ada" : AdaBoostClassifier()}

    models = []
    if parameters == None:
        parameters = {}
    for method in methods:
        if method == "RF":
            if method not in parameters:
                parameter = {}
            elif parameters[method] == "default":
                parameter = {}
            elif parameters[method] == "recommanded":
                parameter = {'max_depth': 10, 'min_samples_leaf': 8, 'min_samples_split': 16, 'n_estimators': 200}
            else:
                parameter = parameters[method]
            model = RandomForestClassifier(**parameter)

        elif method == "LGBM":
            if method not in parameters:
                parameter = {}
            elif parameters[method] == "default":
                parameter = {}
            elif parameters[method] == "recommanded":
                parameter = {'n_esimators' : 200}
            else:
                parameter = parameters[method]
            model = LGBMClassifier(**parameter)

        elif method == "XGB":
            if method not in parameters:
                parameter = {}
            elif parameters[method] == "default":
                parameter = {}
            elif parameters[method] == "recommanded":
                parameter = {'colsample_bytree': 0.5, 'n_estimators': 200, 'subsample': 0.75}
            else:
                parameter = parameters[method]
            model = XGBClassifier(**parameter)

        elif method == "EXtra":
            if method not in parameters:
                parameter = {}
            elif parameters[method] == "default":
                parameter = {}
            elif parameters[method] == "recommanded":
                parameter = {'criterion': 'gini', 'max_features': 'log2', 'min_samples_split': 4, 'n_estimators': 500}
            else:
                parameter = parameters[method]
            model = CatBoostClassifier(**parameter)

        elif method == "Ada":
            if method not in parameters:
                parameter = {}
            elif parameters[method] == "default":
                parameter = {}
            elif parameters[method] == "recommanded":
                parameter = {'algorithm': 'SAMME', 'learning_rate': 1, 'n_estimators': 200}
            else:
                parameter = parameters[method]
            model = AdaBoostClassifier(**parameter)    

        else:
            raise NameError('Error')

        models.append(model)

    X_features = data.iloc[:, :-1]
    y_target = data.iloc[:, -1]
    X_train, X_test, Y_train, Y_test = train_test_split(X_features, y_target, test_size=0.2, random_state=97, stratify=y_target)

    ranking_df = pd.DataFrame()
    importance_df = pd.DataFrame()
    for a, b in zip(models, methods):
        f = a.fit(X_train, Y_train)
        importance = f.feature_importances_

        top_biomarker = pd.DataFrame(importance, index=X_train.columns, columns=['importances_{0}'.format(b)])
        importance_df = pd.concat([importance_df, top_biomarker], axis=1)
    
        ranking = pd.DataFrame()
        ranking[f'ranking_{b}'] = importance_df[f'importances_{b}'].rank(method='min', ascending=False)
        ranking = ranking[[f'ranking_{b}']].astype('int')
        ranking_df = pd.concat([ranking_df, ranking], axis=1)

    return ranking_df, '-'*100, importance_df

