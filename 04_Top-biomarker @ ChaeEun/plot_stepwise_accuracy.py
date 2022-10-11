import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder


def plot_stepwise_accuracy(cancer_df, ranking_df, model, step_num, metric, multi_class):
    prePath = args.PREPATH
    current_path = os.getcwd()
    
    try: 
        os.makedirs(f'{prePath}') 
    except OSError: 
        if not os.path.isdir(f'{prePath}'): 
            raise   
            
    data = pd.read_csv(f'{cancer_df}', index_col='Unnamed: 0', skiprows=[1])
    data['Target'] = LabelEncoder().fit_transform(data['Target'])
    
    ranking_df = pd.read_csv(f'{ranking_df}', index_col='Unnamed: 0')
    
    for method in model:
        if method == 'RF':
            model = RandomForestClassifier()
        elif method == 'MLP':
            model = MLPClassifier()
            
    score_df = pd.DataFrame() 
    methods = ranking_df.columns
    
    for method in methods:
        # step wise
        step_df = pd.DataFrame()
        for num in range(1, step_num):
            top_marker = ranking_df.sort_values(by = method).iloc[:num].index
            # Train, test set split
            feature = data.loc[:, top_marker]
            target = data.iloc[:,-1]
            
            # accuracy_metric
            f1, acc, pre, recall, roc, aic, bic = [], [], [], [], [], [], 
            
            # Stratified-5Fold Training
            skf = StratifiedKFold(n_splits = 5)
            for train_idx, test_idx, in skf.split(feature, target):
                x_train, x_test = feature.iloc[train_idx], feature.iloc[test_idx]
                y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]
                model.fit(x_train, y_train)
                
                # test predict
                y_pred = model.predict(x_test)
                y_proba = model.predict_proba(x_test)
                
                # accuracy_metric
                if multi_class == True:
                    f1.append(metrics.f1_score(y_test, y_pred, average='macro'))
                    acc.append(metrics.accuracy_score(y_test, y_pred))
                    pre.append(metrics.precision_score(y_test, y_pred, average='macro', labels=np.unique(y_pred)))
                    recall.append(metrics.recall_score(y_test, y_pred, average='macro'))
                    roc.append(metrics.roc_auc_score(y_test, y_proba, multi_class='ovo')) 
                else:
                    f1.append(metrics.f1_score(y_test, y_pred))
                    acc.append(metrics.accuracy_score(y_test, y_pred))
                    pre.append(metrics.precision_score(y_test, y_pred))
                    recall.append(metrics.recall_score(y_test, y_pred))
                    roc.append(metrics.roc_auc_score(y_test, y_pred))
                    aic.append(2*metrics.log_loss(y_test, y_proba) + 2*num)
                    bic.append(2*metrics.log_loss(y_test, y_proba) + np.log(x_test.shape[0])*num)   
                    
                if multi_class == True:
                    mean_list, cols = [np.mean(f1), np.mean(acc), np.mean(pre), np.mean(recall), np.mean(roc)], ['f1', 'accuracy', 'precision', 'recall', 'roc']
                else:
                    mean_list, cols = [np.mean(f1), np.mean(acc), np.mean(pre), np.mean(recall), np.mean(roc), np.mean(aic), np.mean(bic)], ['f1', 'accuracy', 'precision', 'recall', 'roc', 'aic', 'bic']
            
            score_step = pd.DataFrame([mean_list], columns=cols)     
            step_df = pd.concat([step_df, score_step[metric]])
        step_df.columns  = [f'{method}_{i}' for i in metric]
        score_df = pd.concat([score_df, step_df], axis=1) 
    score_df = score_df.set_index(pd.Index(range(1,step_num)))
    
    # plot
    fig = plt.figure(figsize=(20,4*len(metric)))
    plt.suptitle(f"Accuracy by step using {model} model", fontsize=30, position = (0.5, 0.95))
    
    for idx, score in enumerate(metric):
        axes = fig.add_subplot(len(metric), 1, idx+1)
        for idx, method in enumerate(methods):
            data = score_df[[i for i in score_df.columns if score in i]]
            axes.plot(range(1,step_num), data.loc[:,f'{method}_{score}'],label = f'{method}', color = sns.color_palette('hsv', len(methods))[idx])
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
    
    os.chdir(f'{prePath}')
    plt.savefig('accuracy.jpeg')
    os.chdir(f'{current_path}')

    os.chdir(f'{prePath}')
    score_df.to_csv('score.csv')
    os.chdir(f'{current_path}')
    
    return score_df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Gene Marker Detection')
    parser.add_argument('--cancer_df', type=str, help='Cancer data', required=True)
    parser.add_argument('--ranking_df', type=str, help='Ranking data', required=True)
    parser.add_argument('--model', nargs='+', help='Model to evaluate performance', choices=['RF','MLP'], default=['RF'])
    parser.add_argument('--step_num', type=int, help='Check performance by number of features', default=10)
    parser.add_argument('--metric', nargs='+', help='To select acuity metrics', default=['acc'], choices=['f1', 'accuracy', 'precision', 'recall', 'roc', 'aic', 'bic'])
    parser.add_argument('--multi_class', type=str, help='Whether to check multi-class', default=None)           
    parser.add_argument('-prePath', help='path of output file', default='./result', metavar='PREPATH', dest='PREPATH')
    
    args = parser.parse_args()
    
    score_df = plot_stepwise_accuracy(args.cancer_df, args.ranking_df, args.model, args.step_num, args.metric, args.multi_class)
    print('------------accuracy-----------')
    print(score_df)
