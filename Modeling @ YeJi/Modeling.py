import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

def modeling_run(data, method):
    model_name =  {"RF" : RandomForestClassifier(),
                 "LGBM" : LGBMClassifier(),
                 "XGB" : XGBClassifier(),
                 "CAT" : CatBoostClassifier(),
                 "Ada" : AdaBoostClassifier()}

    model_final = [keys for keys in method]

    X_features = data.iloc[:, :-1]
    y_target = data.iloc[:, -1]
    X_train, X_test, Y_train, Y_test = train_test_split(X_features, y_target, 
                                                      test_size=0.2, random_state=97, 
                                                      stratify=y_target)
    modeling_df = pd.DataFrame()
    for model in model_final:
        model_run = model_name[model]
        model_run.fit(X_train, Y_train)
    importances = model_run.feature_importances_

    top_biomarker = pd.DataFrame(importances, index=X_train.columns, columns=["{}".format(model)])
    modeling_df = pd.concat([modeling_df, top_biomarker], axis=1)

return modeling_df
