# get_ipython().system('pip install pandas')
# get_ipython().system('pip install sklearn')
# get_ipython().system('pip install seaborn')
# get_ipython().system('pip install matplotlib')
# get_ipython().system('pip install xgboost')

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
#model
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier

def parameter_model(model_param=None):
    model_final = [] 
    model_name = []
    
    for model in model_param:
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

        elif method == 'MLP':
            if parameter == "default":
                param = {}  
            elif parameter == "recommended":
                param = {'activation': 'identity', 'alpha': 0.001, 'hidden_layer_sizes': (400,), 'learning_rate': 'invscaling', 'max_iter': 3000, 'solver': 'adam'}
            else:
                parameter = params
            model = MLPClassifier(**param)
            
        else:
            raise NameError('Error')
        
        model_final.append(model)
        model_name.append(method)

    return model_final, model_name

def plot_stepwise_accuracy(df, ranking_df, step_num, model, accuracy_metric, multi_class=None):
    # top biomaker step
    score_df = pd.DataFrame() 
    methods = ranking_df.columns

    for method in methods:
        # model parameter 
        model_params, model_name = parameter_model([model])
        model_final = model_params.pop() 
        model_name = model_name.pop()
        # step wise
        step_df = pd.DataFrame()
        for num in step_num:
            top_marker = ranking_df.sort_values(by = method).iloc[:num].index
            # Train, test set split
            feature = df.loc[:, top_marker]
            target = df.iloc[:,-1]
            # accuracy_metric
            f1, acc, pre, recall, roc, aic, bic = [], [], [], [], [], [], []
            # Stratified-5Fold Training
            skf = StratifiedKFold(n_splits = 5)
            for train_idx, test_idx, in skf.split(feature, target):
                x_train, x_test = feature.iloc[train_idx], feature.iloc[test_idx]
                y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]
                model_final.fit(x_train, y_train)
                # test predict
                y_pred = model_final.predict(x_test)
                y_proba = model_final.predict_proba(x_test)
                # accuracy_metric
                if multi_class == True:
                    acc.append(metrics.accuracy_score(y_test, y_pred))
                    pre.append(metrics.precision_score(y_test, y_pred))
                    f1.append(metrics.f1_score(y_test, y_pred))
                    recall.append(metrics.recall_score(y_test, y_pred))
                else:
                    f1.append(metrics.f1_score(y_test, y_pred))
                    acc.append(metrics.accuracy_score(y_test, y_pred))
                    pre.append(metrics.precision_score(y_test, y_pred))
                    recall.append(metrics.recall_score(y_test, y_pred))
                    roc.append(metrics.roc_auc_score(y_test, y_pred))
                    aic.append(2*metrics.log_loss(y_test, y_proba) + 2*num)
                    bic.append(2*metrics.log_loss(y_test, y_proba) + np.log(x_test.shape[0])*num)   
                if multi_class == True:
                    mean_list, cols = [np.mean(f1), np.mean(acc), np.mean(pre), np.mean(recall)], ['f1', 'accuracy', 'precision', 'recall']
                else:
                    mean_list, cols = [np.mean(f1), np.mean(acc), np.mean(pre), np.mean(recall), np.mean(roc), np.mean(aic), np.mean(bic)], ['f1', 'accuracy', 'precision', 'recall', 'roc', 'aic', 'bic']
            score_step = pd.DataFrame([mean_list], columns=cols)     
            step_df = pd.concat([step_df, score_step[accuracy_metric]])
        step_df.columns  = [f'{method}_{i}' for i in accuracy_metric]
        score_df = pd.concat([score_df, step_df], axis=1)
    score_df = score_df.set_index(pd.Index(step_num))
    # plot
    fig = plt.figure(figsize=(20,4*len(accuracy_metric)))
    plt.suptitle(f"Accuracy by step using {model_name} model", fontsize=30, position = (0.5, 0.95))
    for idx, score in enumerate(accuracy_metric):
        axes = fig.add_subplot(len(accuracy_metric), 1, idx+1)
        for idx, method in enumerate(methods):
            data = score_df[[i for i in score_df.columns if score in i]]
            axes.plot(step_num, data.loc[:,f'{method}_{score}'],label = f'{method}', color = sns.color_palette('hsv', len(methods))[idx])
            axes.set_xlabel('Step', fontsize=14)
            axes.set_ylabel(f'{score}', fontsize=20)
            # high accuracy
            max_idx = data[data.loc[:,f'{method}_{score}'] == data.loc[:,f'{method}_{score}'].max()].index
            for i in list(max_idx):
                x = i
                y = data.loc[i, f'{method}_{score}']
                axes.text(x, y-0.001, f'Step: {x}\n {np.round(y,3)}', horizontalalignment='center', verticalalignment='top')
                axes.scatter(x, y, color='red')
            plt.legend(loc='lower right')
    plt.show()
    return score_df
