import requests
import os
import shutil
import pandas as pd
import numpy as np
import tarfile
import time
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

def download_data(cancer_list, data_source, data_dir=None):
    current_path = os.getcwd()
    if data_dir is None:
        path = current_path
    else:
        path = f'{current_path}/{data_dir}'

    data = pd.read_csv(data_source)
    for cancer in cancer_list:
        link_df = data[data['cancer'] == cancer]
        link = ''.join(link_df['link'])  
        local_filename = link.split('/')[-1]
        with requests.get(link, stream=True) as r:
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    shutil.copyfileobj(r.raw, f)
                    f.close()
                time.sleep(3)
       
        fname = f'./{local_filename}' 
        ap = tarfile.open(fname)      
        ap.extractall(path)
        ap.close()      
        f = local_filename.rstrip('.tar.gz')
        entries = os.listdir(f'{path}/{f}')
        for entry in entries:
            if cancer in entry:
                shutil.move(f'{path}/{f}/{entry}', path)
                shutil.rmtree(f'{path}/{f}')
        os.remove(f'{fname}')   


def load_labeled_data(data_dir, label_list, patient_type=None):
    file_list = [f for f in os.listdir(data_dir) if f != '.ipynb_checkpoints']

    cancer_df = pd.DataFrame()
    for idx, label in enumerate(label_list):
        cancer = label.split('_')[1]
        cancer_txt = "".join([f for f in file_list if f.split('.')[0] == cancer])
        data = pd.read_csv(f'{data_dir}/{cancer_txt}', sep='\t', low_memory=False, index_col='Hybridization REF', skiprows=[1])
        data = data.transpose()
        
        if cancer == 'BRCA':
            subtype = label.split('_')[2]
            sub_file = pd.read_csv(patient_type, sep=',', index_col='Hybridization REF')
            BRCA = pd.merge(data, sub_file, left_index=True, right_index=True, how='left')
            BRCA = BRCA[(BRCA['Tumor'] == 'BRCA')&(BRCA['Subtype'].str.lower() == subtype.lower())]
            data = BRCA.drop(columns = ['Tumor'])
            
        data['Target'] = data.index.str[13:15]
        data['Target'].replace(['01', '02', '03', '04', '05', '06', '07', '08', '09'], 'Cancer', inplace=True)
        data['Target'].replace(['10', '11', '12', '13', '14', '15', '16', '17', '18', '19'],'Normal', inplace=True)
        data = data[data['Target'].str.contains('Cancer')]

        data['Target'] = label
        cancer_df = pd.concat([cancer_df, data])
        
    return cancer_df


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
                param = {'max_depth': 1, 'min_samples_leaf': 1, 'min_weight_fraction_leaf': 0.0}
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
    ranking = pd.DataFrame()
    for modeling, method_name in zip(models_final, models):
        f_importance = modeling.fit(X_train, Y_train)
        importance = f_importance.feature_importances_    
     
        biomarker_importance = pd.DataFrame(importance, index=X_train.columns, columns=[f'{method_name[0]}'])
        importance_df = pd.concat([importance_df, biomarker_importance], axis=1)

        ranking[f'{method_name[0]}'] = importance_df[f'{method_name[0]}'].rank(method='min', ascending=False)
        ranking = ranking[[f'{method_name[0]}']].astype('int')
        ranking_df = pd.concat([ranking_df, ranking], axis=1)

    return ranking_df, importance_df


def parameter_model(model_param=None):
    method = ''.join(model_param.keys())
    params = model_param[method]

    if method == 'RF':
        if params == "default":
            parameter = {}  
        elif params == "recommended":
            parameter = {'max_depth': 10, 'min_samples_leaf': 8, 'min_samples_split': 16, 'n_estimators': 200}
        else:
            parameter = params
        model = RandomForestClassifier(**parameter)

    elif method == 'MLP':
        if params == "default":
            parameter = {}  
        elif params == "recommended":
            parameter = {'activation': 'identity', 'alpha': 0.001, 'hidden_layer_sizes': (400,), 'learning_rate': 'invscaling', 'max_iter': 3000, 'solver': 'adam'}
        else:
            parameter = params
        model = MLPClassifier(**parameter)

    return model, method

def plot_stepwise_accuracy(df, ranking_df, step_num, model, accuracy_metric):
    model, model_name = parameter_model(model)

    score_df = pd.DataFrame() 
    methods = ranking_df.columns
    step_df = pd.DataFrame()
    for method in methods:
        step_df = pd.DataFrame()
        for num in step_num:
            top_marker = ranking_df.sort_values(by = method).iloc[:num].index
            feature = df.loc[:, top_marker]
            target = df.iloc[:,-1]
            x_train, x_test, y_train, y_test = train_test_split(feature, target, test_size=0.2, stratify=target, random_state=20) #class 비율 유지
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)

            score_list = []
            for idx, metric in enumerate(accuracy_metric):
                if metric == 'f1':
                    score = metrics.f1_score(y_test, y_pred)
                elif metric == 'accuracy':
                    score = metrics.accuracy_score(y_test, y_pred)
                elif metric == 'precision':
                    score = metrics.precision_score(y_test, y_pred)
                elif metric == 'recall':
                    score = metrics.recall_score(y_test, y_pred)
                elif metric == 'roc':
                    score = metrics.roc_auc_score(y_test, y_pred)
                score_list.append(score)
            data = pd.DataFrame([score_list], columns = [f'{method}_{i}' for i in accuracy_metric])
            step_df = pd.concat([step_df, data])
        score_df = pd.concat([score_df, step_df], axis=1)
    score_df = score_df.set_index(pd.Index(step_num))

    for score in accuracy_metric:
        fig , ax = plt.subplots(figsize=(15,5))
        data = score_df[[i for i in score_df.columns if score in i]]
        plt.suptitle(f"{model_name} TopMaker {score}", fontsize=15)
        ax.set_ylim([data.iloc[:,:].min().min() - 0.03 , data.iloc[:,:].max().max() + 0.03])
        ax.set_ylabel(f'{score}', fontsize=12)
        ax.set_xlabel('Step', fontsize=12)
        for idx, method in enumerate(methods):
            ax.plot(step_num, data.iloc[:,idx],label = f'{method}')
            max_idx = data[data.iloc[:,idx] == data.iloc[:,idx].max()].index
            for i in list(max_idx):
                x = i
                y = data.loc[i, f'{method}_{score}']
                ax.text(x, y+0.001, f'Step: {x}\n {np.round(y,3)}', horizontalalignment='center', verticalalignment='bottom')
            ax.scatter(x, y, color='red')
            plt.legend(loc='lower right')
        plt.show()
    return score_df


def normalize(df, methods):
    normalize_df = df.copy()
    
    for method in methods:
        if method == 'log1p':
            for col in df.columns:
                normalize_df.loc[:,col] = np.log1p(df.loc[:,col])
        
        if method == 'z_score':
            for col in normalize_df.columns:
                m, s = normalize_df.loc[:,col].mean(), normalize_df.loc[:,col].std()
                if s == 0.0:
                    normalize_df.loc[:,col] = 0.0
                else:
                    normalize_df.loc[:,col] = (normalize_df.loc[:,col] - m)/s  
    return normalize_df

def plot_heatmap(df):
    df = df.transpose()    
    dmin, dmax = df.min(), df.max()
    fig, ax = plt.subplots(figsize=(10,7))
    sns.heatmap(df, 
                  cmap = 'RdBu_r',
                  cbar_kws = {"shrink": .8},
                  center = 0.0,
                  vmin = -2,
                  vmax = 2,
                  xticklabels=False
                )
    ax.set_xticks([len(df.columns)/4,len(df.columns)*3/4])
    ax.set_xticklabels(['LUAD','LUSC'])
    ax.set_title('LUAD/LUSC')
    ax.set_xlabel('Patients')
    ax.set_ylabel('Genome')
    plt.show()  
