get_ipython().system('pip install pandas')
get_ipython().system('pip install sklearn')
get_ipython().system('pip install lightgbm')
get_ipython().system('pip install xgboost')

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

def biomarker_rank(data, models):
    models_final = []
    for model in models:
        method = model[0]
        parameter = model[1]
    
        if method == 'RF':
            if parameter == 'default':
                param = {}
            elif parameter == 'recommended' :
                param = {'max_depth': 10, 'min_samples_leaf': 8, 'min_samples_split': 16, 'n_estimators': 200}
            else :
                param = parameter
            model = RandomForestClassifier(**param)
     
        elif method == 'XGB':
            if parameter == 'default':
                param = {}
            elif parameter == 'recommended' :
                param = {'colsample_bytree': 0.5, 'n_estimators': 200, 'subsample': 0.75}
            else :
                param = parameter
            model = XGBClassifier(**param)

        elif method == 'EXtra':
            if parameter == 'default':
                param = {}
            elif parameter == 'recommended' :
                param = {'criterion': 'gini', 'max_features': 'log2', 'min_samples_split': 4, 'n_estimators': 500}
            else :
                param = parameter
            model = ExtraTreesClassifier(**param)

        elif method == 'Ada':
            if parameter == 'default':
                param = {}
            elif parameter == 'recommended' :
                param = {'algorithm': 'SAMME', 'learning_rate': 1, 'n_estimators': 200}
            else :
                param = parameter
            model = AdaBoostClassifier(**param)  

        elif method == 'DT':
            if parameter == 'default':
                param = {}
            elif parameter == 'recommended' :
                param = {}
            else :
                param = parameter
            model = DecisionTreeClassifier(**param) 

        else:
            raise NameError('Error')

        models_final.append(model)    

    X_features = data.iloc[:, :-1]
    y_target = data.iloc[:, -1]
    X_train, X_test, Y_train, Y_test = train_test_split(X_features, y_target, test_size=0.2, random_state=97, stratify=y_target)

    ranking_df = pd.DataFrame()
    importance_df = pd.DataFrame()  
    for modeling, method_name in zip(models_final, models):
        f_importance = modeling.fit(X_train, Y_train)
        importance = f_importance.feature_importances_    
     
        biomarker_importance = pd.DataFrame(importance, index=X_train.columns, columns=[f'{method_name[0]}'])
        importance_df = pd.concat([importance_df, biomarker_importance], axis=1)

        ranking = pd.DataFrame()
        ranking[f'{method_name[0]}'] = importance_df[f'{method_name[0]}'].rank(method='min', ascending=False)
        ranking = ranking[[f'{method_name[0]}']].astype('int')
        ranking_df = pd.concat([ranking_df, ranking], axis=1)

    return ranking_df, importance_df

