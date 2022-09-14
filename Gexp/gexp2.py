import numpy as np
import os
import pip

try:
    import seaborn as sns
except ImportError:
    get_ipython().system('pip install seaborn')
finally:
    import seaborn as sns

try:
    import matplotlib.pyplot as plt 
except ImportError:
    get_ipython().system('pip install matplotlib')
finally:
    import matplotlib.pyplot as plt 

try:
    import pandas as pd
except ImportError:
    get_ipython().system('pip install pandas')
finally:
    import pandas as pd
    
try:
    import argparse 
except ImportError:
    get_ipython().system('pip install argparse')
finally:
    import argparse
    
try:
    from xgboost import XGBClassifier
except ImportError:
    get_ipython().system('pip install xgboost')
finally:
    from xgboost import XGBClassifier

try:
    from sklearn import metrics
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import LabelEncoder
except ImportError:
    get_ipython().system('pip install sklearn')
finally:
    from sklearn import metrics
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import LabelEncoder
    

def gexp(data_path, cancers, models, topN, output_path):
    current_path = os.getcwd()
    try: 
        os.makedirs(f'{output_path}') 
    except OSError: 
        if not os.path.isdir(f'{output_path}'): 
            raise
            
    df = pd.DataFrame()
    for cancer in cancers:
        data = pd.read_csv(f"{data_path}/{cancer}.rnaseqv2__illuminahiseq_rnaseqv2__unc_edu__Level_3__RSEM_genes_normalized__data.data.txt", sep='\t', low_memory=False, index_col='Hybridization REF', skiprows=[1])
        data = data.transpose() 

        data['Target'] = data.index.str[13:15]
        data['Target'].replace(['01', '02', '03', '04', '05', '06', '07', '08', '09'], 'Cancer', inplace=True)
        data['Target'].replace(['10', '11', '12', '13', '14', '15', '16', '17', '18', '19'],'Normal', inplace=True)
        data = data[data['Target'].str.contains('Cancer')]
        
        data['Target'] = cancer
        df = pd.concat([df, data])
        
    data = df.copy()
    df['Target'] = LabelEncoder().fit_transform(df['Target'])
    
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

    X_features = df.iloc[:, :-1]
    Y_target = df.iloc[:, -1]
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

    ranking_sort = ranking_df.sort_values(by=models, ascending=True)
    
    if topN!=10000000000:
        ranking_top = ranking_sort.iloc[:topN,:]
        
    os.chdir(f'{output_path}')
    ranking_top.to_csv('rank.csv')
    os.chdir(f'{current_path}')
    
    result_df = pd.DataFrame()
    for name in models:
        score_step = pd.DataFrame()
        for num in np.arange(1,11):
            top_marker = ranking_df.sort_values(by = name).iloc[:num].index

            x_feature = df.loc[:, top_marker]
            y_target = df.iloc[:,-1]   

            skf = StratifiedKFold(n_splits = 5)
            for train_idx, test_idx, in skf.split(x_feature, y_target):
                x_train, x_test = x_feature.iloc[train_idx], x_feature.iloc[test_idx]
                y_train, y_test = y_target.iloc[train_idx], y_target.iloc[test_idx]
                performance = RandomForestClassifier()
                performance.fit(x_train, y_train)

                y_pred = performance.predict(x_test)

                acc = []
                acc.append(metrics.accuracy_score(y_test, y_pred))

                mean_list = [np.mean(acc)]
                score_df = pd.DataFrame([mean_list])

            score_step = pd.concat([score_step, score_df])
            score_step.reset_index(drop=True, inplace=True)

        result_df = pd.concat([result_df, score_step], axis=1)
    result_df.columns = [f'{name}' for name in models]
    result_df.index = np.arange(1,11)

    plt.figure(figsize=(10,5))
    for idx,m in enumerate(models):
        plt.plot(np.arange(1,11), result_df[m], label=f'{m}',color=sns.color_palette('hsv', len(models))[idx])

        argmx = result_df[m].argmax()+1
        mn, mx = result_df.loc[:,m].min(), result_df.loc[:,m].max()
        result_df.loc[:,m].min()
        plt.scatter(argmx, mx, color='red')
        plt.text(argmx, mx-0.002, f'Step: {argmx}\n {np.round(mx,3)}', horizontalalignment='center', verticalalignment='top')

        plt.title(f'Accuracy by step using models', fontsize=20)
        plt.xticks(np.arange(1,10,1))
        plt.xlabel("Step", fontsize=10)
        plt.ylabel("Accuracy", fontsize=10)
        plt.legend(loc='lower right')
    
    os.chdir(f'{output_path}')
    plt.savefig('accuracy.jpeg')
    os.chdir(f'{current_path}')

    normalize_df = data.copy()
    normalize_df.drop(columns = 'Target', inplace=True)
    for col in normalize_df.columns:
        normalize_df.loc[:,col] = np.log1p(normalize_df.loc[:,col])

        m, s = normalize_df.loc[:,col].mean(), normalize_df.loc[:,col].std()
        if s == 0.0:
            normalize_df.loc[:,col] = 0.0
        else:
            normalize_df.loc[:,col] = (normalize_df.loc[:,col] - m)/s

    normalize_df = pd.concat([normalize_df, data[['Target']]], axis=1)

    cancer_list = list(normalize_df['Target'])
    ind = sorted(np.unique(cancer_list, return_index=True)[1])
    names = [cancer_list[i] for i in sorted(ind)]

    pal = sns.husl_palette(len(names), s=.45)
    lut = dict(zip(map(str, np.unique(cancer_list)),pal))
    cancer_colors = np.asarray(pd.Series(cancer_list).map(lut))

    gene_list = ranking_df.sort_values(by='RF').iloc[:topN].index
    hmap_data = normalize_df.loc[:,gene_list].transpose()
    cmap = sns.clustermap(hmap_data, 
                  cmap = 'RdBu_r',
                  cbar_kws = {"shrink": .8},
                  center = 0.0,
                  vmin = -2,
                  vmax = 2,
                  linewidths=0,
                  xticklabels=False,
                  cbar_pos=(1, .015, .04, .75),
                  row_cluster=True, 
                  col_cluster=False,
                  col_colors= cancer_colors
                )

    ind.append(len(cancer_list))
    cmap.ax_heatmap.set_xticks([((ind[i]+ind[i+1])/2) for i in range(len(ind)-1)])
    cmap.ax_heatmap.set_xticklabels(names)
    cmap.ax_heatmap.set_xlabel(' / '.join([i for i in names]), fontsize = 15, labelpad = 25)
    cmap.ax_heatmap.set_title('cancer biomaker heatmap', y = 1.1, fontsize = 25)

    os.chdir(f'{output_path}')
    plt.savefig('heatmap.jpeg')
    os.chdir(f'{current_path}')
        
    return ranking_df, result_df
    
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Gene Marker Detection')
    parser.add_argument('--data_path', type=str, help='Path where the data exists', default='./data')
    parser.add_argument('--cancers', nargs='+', help='Data to download and analyze', required=True)
    parser.add_argument('--models', nargs='+', help='Modeling for Featuer Importances', choices=['RF', 'XGB', 'Ada','EXtra','DT'], required=True)
    parser.add_argument('--topN', type=int, help='Adjusting the train/tset Rate ', default=10000000000)
    parser.add_argument('--output_path', type=str, help='Output path to save the result file', default='./result')
    
    args = parser.parse_args()
    
    gexp(args.data_path, args.cancers, args.models, args.topN, args.output_path)
