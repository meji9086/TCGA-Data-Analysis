# get_ipython().system('pip install pandas')
# get_ipython().system('pip install matplotlib')
# get_ipython().system('pip install sklearn')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


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
    # model parameter 
    model, model_name = parameter_model(model)

    # top biomaker step
    score_df = pd.DataFrame() 
    methods = ranking_df.columns
    step_df = pd.DataFrame()
    for method in methods:
        step_df = pd.DataFrame()
        for num in step_num:
            top_marker = ranking_df.sort_values(by = method).iloc[:num].index
            feature = df.loc[:, top_marker]
            target = df.iloc[:,-1]
            # model training
            x_train, x_test, y_train, y_test = train_test_split(feature, target, test_size=0.2, stratify=target, random_state=20) #class 비율 유지
            model.fit(x_train, y_train)

            # model test
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

    # Plot Accuracy
    for score in accuracy_metric:
      fig , ax = plt.subplots(figsize=(15,5))
      data = score_df[[i for i in score_df.columns if score in i]]
      plt.suptitle(f"{model_name} TopMaker {score}", fontsize=15)
      ax.set_ylim([data.iloc[:,:].min().min() - 0.03 , data.iloc[:,:].max().max() + 0.03])
      ax.set_ylabel(f'{score}', fontsize=12)
      ax.set_xlabel('Step', fontsize=12)
      for idx, method in enumerate(methods):
          ax.plot(step_num, data.iloc[:,idx],label = f'{method}')
          # high accuracy
          max_idx = data[data.iloc[:,idx] == data.iloc[:,idx].max()].index
          for i in list(max_idx):
            x = i
            y = data.loc[i, f'{method}_{score}']
            ax.text(x, y+0.001, f'Step: {x}\n {np.round(y,3)}', horizontalalignment='center', verticalalignment='bottom')
          ax.scatter(x, y, color='red')
          plt.legend(loc='lower right')
      plt.show()
    return score_df