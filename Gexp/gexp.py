import os
import pip
import time
import numpy as np
import gzip
from IPython.display import HTML

try:
    import requests
except ImportError:
    pget_ipython().system('pip install requests')
finally:
    import requests

try:
    import pandas
except ImportError:
    get_ipython().system('pip install pandas')
finally:
    import pandas

try:
    import shutil
except ImportError:
    get_ipython().system('pip install shutil')
finally:
    import shutil

try:
    import tarfile
except ImportError:
    get_ipython().system('pip install tarfile')
finally:
    import tarfile

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
    from sklearn import metrics
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import StratifiedKFold
except ImportError:
    get_ipython().system('pip install sklearn')
finally:
    from sklearn import metrics
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import StratifiedKFold

try:
    from xgboost import XGBClassifier
except ImportError:
    get_ipython().system('pip install xgboost')
finally:
    from xgboost import XGBClassifier
    
def download_metadata(metadata_dir='./Metadata/'):
    url = 'https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz'
    file_name = url.split('/')[-1]
    with open(file_name, "wb") as file:
        response = requests.get(url)               
        file.write(response.content)
    while True:
        try:
            if file_name in os.listdir(os.getcwd()):
                os.makedirs(metadata_dir)
                shutil.move(file_name, f'{metadata_dir}{file_name}')
                break
        except:
            pass
    f = gzip.open(f'{metadata_dir}{file_name}', 'rb')
    file_content = f.read()
    with open(f"{metadata_dir}{os.path.splitext(file_name)[0]}", "wb") as f:
        f.write(file_content)
        f.close()
    os.remove(f'{metadata_dir}{file_name}')
    
# Metadata download
download_metadata()    

def download_data(cancer_list, data_source='./cancer_link.csv', data_dir=None):
    current_path = os.getcwd()
    if data_dir is None:
        path = current_path
    else:
        path = f'{current_path}/{data_dir}'

    try: 
        os.makedirs(path) 
    except OSError: 
        if not os.path.isdir(path): 
            raise
            
    down_folder = [f.split(".")[0] for f in os.listdir(path) if f != '.ipynb_checkpoints']
    for cancer in cancer_list:
        if cancer in down_folder:
                print(f'The {cancer} already exists')
                continue       
        else:
            data = pd.read_csv(data_source)
            link = ''.join(data[data['cancer'] == cancer].loc[:,'link'])
            local_filename = link.split('/')[-1]
            try:
                with requests.get(link, stream=True) as r: 
                    with open(local_filename, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                            shutil.copyfileobj(r.raw, f)
                            f.close() 

                if local_filename in os.listdir(current_path):
                    fname = f'./{local_filename}'   
                    ap = tarfile.open(fname)    
                    ap.extractall(path)
                    ap.close()

                    folder = local_filename.rstrip('.tar.gz')
                    entries = os.listdir(f'{path}/{folder}')
                    for entry in entries:
                        if cancer in entry:
                            shutil.move(f'{path}/{folder}/{entry}', path)
                            shutil.rmtree(f'{path}/{folder}')
                    os.remove(f'{fname}') 
                else:
                     break
                            
            except:
                time.sleep(1)    
            print(f'The {cancer} is downloaded successfully')

def load_labeled_data(data_dir, label_list, patient_type=None):
    # file list
    file_list = [f for f in os.listdir(data_dir) if f != '.ipynb_checkpoints']
    cancer_df = pd.DataFrame()
    for idx, label in enumerate(label_list):
        cancer = label.split('_')[0]
        cancer_txt = "".join([f for f in file_list if f.split('.')[0] == cancer])
        data = pd.read_csv(f'{data_dir}/{cancer_txt}', sep='\t', low_memory=False, index_col='Hybridization REF', skiprows=[1])
        data = data.transpose()
        
        data['Target'] = data.index.str[13:15]
        data['Target'].replace(['01', '02', '03', '04', '05', '06', '07', '08', '09'], 'Cancer', inplace=True)
        data['Target'].replace(['10', '11', '12', '13', '14', '15', '16', '17', '18', '19'],'Normal', inplace=True)
        data = data[data['Target'].str.contains('Cancer')]
 
        if cancer == 'BRCA':
            subtype = label.split('_')[1]
            sub_file = pd.read_csv(patient_type, sep=',', index_col='Hybridization REF')
            BRCA = pd.merge(data, sub_file, left_index=True, right_index=True, how='left')
            BRCA = BRCA[(BRCA['Tumor'] == 'BRCA')&(BRCA['Subtype'].str.lower() == subtype.lower())]
            data = BRCA.drop(columns = ['Tumor', 'Subtype'])

        data['Target'] = label
        cancer_df = pd.concat([cancer_df, data])
        print(f'load file : {cancer_txt}')
        
    return cancer_df

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

def biomarker_rank(data, models=[['RF','recommended']], test_size=0.2):
    model_final, model_name = parameter_model(models)

    X_features = data.iloc[:, :-1]
    y_target = data.iloc[:, -1]
    X_train, X_test, Y_train, Y_test = train_test_split(X_features, y_target, test_size=0.2, random_state=97, stratify=y_target)

    ranking_df = pd.DataFrame()
    importance_df = pd.DataFrame()  
    for modeling, method_name in zip(model_final, model_name):
        f_importance = modeling.fit(X_train, Y_train)
        importance = f_importance.feature_importances_    

        biomarker_importance = pd.DataFrame(importance, index=X_train.columns, columns=[f'{method_name}'])
        importance_df = pd.concat([importance_df, biomarker_importance], axis=1)

        ranking = pd.DataFrame()
        ranking[f'{method_name}'] = importance_df[f'{method_name}'].rank(method='min', ascending=False)
        ranking = ranking[[f'{method_name}']].astype('int')
        ranking_df = pd.concat([ranking_df, ranking], axis=1)
        
    ranking_df = ranking_df.sort_values(by=model_name[0], ascending=True)
    importance_df = importance_df.sort_values(by=model_name[0], ascending=False)

    return ranking_df, importance_df

def plot_stepwise_accuracy(df, ranking_df, step_num, model=['RF','recommended'], accuracy_metric=['accuracy'], multi_class=None):
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

def normalize(df, methods=['log1p', 'z_score'], exclude='Target'):
    normalize_df = df.copy()
    normalize_df.drop(columns = exclude, inplace=True)

    for method in methods:
        if method == 'log1p':
            for col in normalize_df.columns:
                normalize_df.loc[:,col] = np.log1p(normalize_df.loc[:,col])
        
        if method == 'z_score':
            for col in normalize_df.columns:
                m, s = normalize_df.loc[:,col].mean(), normalize_df.loc[:,col].std()
                if s == 0.0:
                    normalize_df.loc[:,col] = 0.0
                else:
                    normalize_df.loc[:,col] = (normalize_df.loc[:,col] - m)/s  
    normalize_df = pd.concat([normalize_df, df[[exclude]]], axis=1)
    return normalize_df

def describe_genes(gene_list, gene_description='./Metadata/Homo_sapiens.gene_info'):
    df = pd.read_csv(gene_description, sep='\t')
    df = df[['GeneID', 'Symbol', 'type_of_gene', 'map_location', 'description']]
    
    df2 = pd.DataFrame(data=None, columns=df.columns)
    for gene in gene_list:
        gsymbol, gid = gene.split('|')[0], int(gene.split('|')[1])
        if sum(df['GeneID'] == gid) == 1:
            df2 = pd.concat([df2,df.loc[df['GeneID'] == gid]], ignore_index=True)
        elif sum(df['Symbol'] == gsymbol) == 1:
            df2 = pd.concat([df2,df.loc[df['Symbol'] == gsymbol]], ignore_index=True)
        else:
            df2 = df2.append({'GeneID': gid, 'Symbol': gsymbol, 'type_of_gene': None, 'map_location': None, 'description': None}, ignore_index=True)

    df2['links']= 'https://www.genecards.org/cgi-bin/carddisp.pl?gene='+df2['GeneID'].astype(str)
    df2 = HTML(df2.to_html(render_links=True, escape=False))
    
    return df2

def plot_heatmap(df, gene_list, vmin=-2, vmax=2):
    
    cancer_list = list(df['Target'])
    ind = sorted(np.unique(cancer_list, return_index=True)[1])
    names = [cancer_list[i] for i in sorted(ind)]
    
    pal = sns.husl_palette(len(names), s=.45)
    lut = dict(zip(map(str, np.unique(cancer_list)),pal))
    cancer_colors = np.asarray(pd.Series(cancer_list).map(lut))
    
    data = df.loc[:,gene_list].transpose()
    cmap = sns.clustermap(data, 
                  cmap = 'RdBu_r',
                  cbar_kws = {"shrink": .8},
                  center = 0.0,
                  vmin = vmin,
                  vmax = vmax,
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
