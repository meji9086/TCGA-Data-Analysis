import os
import glob
import numpy as np
import pandas as pd

def Load_Labeling(file, cancer_list):
    download_path = os.getcwd() + f'/{file}'
    path_list = os.listdir(download_path)
    path_list = [x for x in path_list if x != '.ipynb_checkpoints']
    cancer_df = pd.DataFrame()
    for idx, f in enumerate(path_list):
        cancer = cancer_list[idx] # 불러올 암 이름
        cancer_txt = ''.join([f for f in path_list if f[:4] == cancer])
        print(f'cancer : {cancer}\nfile : {cancer_txt}') 
        # file load
        data = pd.read_csv(f'{download_path}/{cancer_txt}', sep='\t', low_memory=False, index_col='Hybridization REF', skiprows=[1])
        data = data.transpose()
        # Separation of cancer patients 
        data['Target'] = data.index.str[13:15]
        data['Target'] = data['Target'].replace({'01':'cancer', '02':'cancer', '11':'Normal', '10':'Normal'})
        data['Target'] = cancer +'_'+ data['Target']
        data = data[data['Target'].str.contains('cancer')]
        cancer_df = pd.concat([cancer_df, data])
    # cancer data labeling
    cancer_df = pd.get_dummies(cancer_df, columns=['Target'], drop_first=True)
    cancer_df.rename(columns = {f'Target_{cancer}_cancer': cancer}, inplace=True)
    return cancer_df