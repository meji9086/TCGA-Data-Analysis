get_ipython().system('pip install pandas')
get_ipython().system('pip install matplotlib')
get_ipython().system('pip install sklearn')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


def Parameter_Model(model_parameter=None):
    
    method = ''.join([i for i in model_parameter.keys()])
    params = model_parameter[method]
    if method == 'RF':
        if params == "default":
            parameter = {}  
        elif params == "recommended":
            parameter = {'max_depth': 10, 'min_samples_leaf': 8, 'min_samples_split': 16}
        else:
            parameter = params
        model = RandomForestClassifier(**parameter)
    elif method == 'MLP':
        if params == "default":
            parameter = {}  
        elif params == "recommended":
            parameter = {'max_iter': 500}
        else:
            parameter = params
        model = MLPClassifier(**parameter)
    return model, method      


def Topbio_test(df, ranking_df, methods, step, model_parameter):
    # model parameter 
    model, model_name = Parameter_Model(model_parameter)
    score_df = pd.DataFrame(step, columns = ['Step']) 
    #top biomaker step
    for method in methods:
        step_df = pd.DataFrame()
        for num in step:
            top_marker = ranking_df.sort_values(by = f'ranking_{method}').iloc[:num].index
            feature = df.loc[:, top_marker]
            target = df.iloc[:,-1]
            # train, test set
            x_train, x_test, y_train, y_test = train_test_split(feature, target, test_size=0.2, stratify=target, random_state=20) #class 비율 유지
            # model training
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            acc = accuracy_score(y_test, y_pred)
            data = pd.DataFrame([[num,acc]], columns = ['Step', f'{method}'])
            step_df = pd.concat([step_df, data]).reset_index(drop=True)
        score_df = pd.merge(step_df, score_df, how='inner', on = 'Step').reset_index(drop=True)

  # Plot Accuracy
    fig , ax = plt.subplots(figsize=(15,5))
    plt.suptitle(f"{model_name} TopMaker Accuracy", fontsize=15)
    ax.set_ylim([score_df.iloc[:, 1:].min().min() - 0.03 , score_df.iloc[:, 1:].max().max() + 0.03])
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_xlabel('Step', fontsize=12)
    for i in methods:
        ax.plot(list(score_df['Step']), score_df[f'{i}'], label = f'{i}')
        # high accuracy
        max_idx = score_df[score_df[f'{i}'] == score_df[f'{i}'].max()].index
        for idx in list(max_idx):      
            x = score_df.loc[idx,'Step']
            y = score_df.loc[idx,f'{i}']
            ax.text(x, y+0.001, f'Step: {x}\n {np.round(y,3)}', horizontalalignment='center', verticalalignment='bottom')
            ax.scatter(x,y, color='red')
    plt.legend(loc='lower right')
    plt.show()
    return score_df