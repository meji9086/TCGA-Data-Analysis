import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

def model_parameter(methods, parameters=None):
    models = {}
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
            model = ExtraTreesClassifier(**parameter)

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

        models[method] = model 
    return models