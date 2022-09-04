#get_ipython().system('pip install pandas')
import os
import pandas as pd

def load_labeled_data(data_dir, label_list, patient_type=None):
    # file list
    file_list = [f for f in os.listdir(data_dir) if f != '.ipynb_checkpoints']
    # data load
    cancer_df = pd.DataFrame()
    for idx, label in enumerate(label_list):
        cancer = label.split('_')[1]
        cancer_txt = "".join([f for f in file_list if f.split('.')[0] == cancer])
        data = pd.read_csv(f'{data_dir}/{cancer_txt}', sep='\t', low_memory=False, index_col='Hybridization REF', skiprows=[1])
        data = data.transpose()
        
        # BRCA data
        if cancer == 'BRCA':
            subtype = label.split('_')[2]
            sub_file = pd.read_csv(patient_type, sep=',', index_col='Hybridization REF')
            BRCA = pd.merge(data, sub_file, left_index=True, right_index=True, how='left')
            BRCA = BRCA[(BRCA['Tumor'] == 'BRCA')&(BRCA['Subtype'].str.lower() == subtype.lower())]
            data = BRCA.drop(columns = ['Tumor'])
            
        # delete normal patients
        data['Target'] = data.index.str[13:15]
        data['Target'].replace(['01', '02', '03', '04', '05', '06', '07', '08', '09'], 'Cancer', inplace=True)
        data['Target'].replace(['10', '11', '12', '13', '14', '15', '16', '17', '18', '19'],'Normal', inplace=True)
        data = data[data['Target'].str.contains('Cancer')]
        #data labeld
        data['Target'] = label
        cancer_df = pd.concat([cancer_df, data])
        
    return cancer_df