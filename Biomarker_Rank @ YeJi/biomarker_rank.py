import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


def biomarker_rank(cancer_df, models):
    prePath = args.PREPATH
    current_path = os.getcwd()
    
    try: 
        os.makedirs(f'{prePath}') 
    except OSError: 
        if not os.path.isdir(f'{prePath}'): 
            raise   
            
    data = pd.read_csv(f'{cancer_df}', index_col='Unnamed: 0', skiprows=[1])        
    data['Target'] = LabelEncoder().fit_transform(data['Target'])
    
    model_final = []
    for method in models:
        if method == 'RF':
            model = RandomForestClassifier()
     
        elif method == 'XGB': 
            model = XGBClassifier()

        elif method == 'EXtra':
            model = ExtraTreesClassifier()

        elif method == 'Ada':
            model = AdaBoostClassifier()

        elif method == 'DT':
            model = DecisionTreeClassifier() 
            
        model_final.append(model)

    X_features = data.iloc[:, :-1]
    Y_target = data.iloc[:, -1]
    X_train, X_test, Y_train, Y_test = train_test_split(X_features, Y_target, test_size=0.2, random_state=97, stratify=Y_target)
    
    ranking_df = pd.DataFrame()
    importance_df = pd.DataFrame()
    for modeling, method_name in zip(model_final, models):
        f_importance = modeling.fit(X_train, Y_train)
        importance = f_importance.feature_importances_    

        biomarker_importance = pd.DataFrame(importance, index=X_train.columns, columns=[f'{method_name}'])
        importance_df = pd.concat([importance_df, biomarker_importance], axis=1)

        ranking = pd.DataFrame()
        ranking[f'{method_name}'] = importance_df[f'{method_name}'].rank(method='min', ascending=False)
        ranking = ranking[[f'{method_name}']].astype('int')
        ranking_df = pd.concat([ranking_df, ranking], axis=1)

    ranking_sort = ranking_df.sort_values(by=models[0], ascending=True)
        
    os.chdir(f'{prePath}')
    ranking_sort.to_csv('rank.csv')
    os.chdir(f'{current_path}')
    
    return ranking_sort
    
    
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Gene Marker Detection')
    parser.add_argument('--cancer_df', type=str, help='Cancer data', required=True)
    parser.add_argument('--models', nargs='+', help='Modeling for Featuer Importances', choices=['RF', 'XGB', 'Ada','EXtra','DT'], default=['RF'])
    parser.add_argument('-prePath', help='path of output file', default='./result', metavar='PREPATH', dest='PREPATH')
    
    args = parser.parse_args()
    
    rank = biomarker_rank(args.cancer_df, args.models)
    print('------------Ranking-----------')
    print(rank)
