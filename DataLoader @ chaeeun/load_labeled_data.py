import os
import pandas as pd

def load_labeled_data(data_dir, label_list, patient_type):
    prePath = args.PREPATH
    current_path = os.getcwd()

    try: 
        os.makedirs(f'{prePath}') 
    except OSError: 
        if not os.path.isdir(f'{prePath}'): 
            raise   

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
            if label == 'BRCA':
                pass
            else:
                subtype = label.split('_')[1]
                sub_file = pd.read_csv(patient_type, sep=',', index_col='Hybridization REF')
                BRCA = pd.merge(data, sub_file, left_index=True, right_index=True, how='left')
                BRCA = BRCA[(BRCA['Tumor'] == 'BRCA')&(BRCA['Subtype'].str.lower() == subtype.lower())]
                data = BRCA.drop(columns = ['Tumor', 'Subtype'])

        data['Target'] = label
        cancer_df = pd.concat([cancer_df, data])
        print(f'load file : {cancer_txt}')
        
    os.chdir(f'{prePath}')
    cancer_df.to_csv('cancer_data.csv')
    os.chdir(f'{current_path}')
    
    return cancer_df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Gene Marker Detection')
    parser.add_argument('--data_dir', type=str, help='Path where the data exists', required=True)
    parser.add_argument('--label_list', nargs='+', help='Name of cancer to label', required=True)
    parser.add_argument('--patient_type', type=str, help='Path where metadata exists', default='./Metadata/BRCApatients_type.csv')
    parser.add_argument('-prePath', help='path of output file', default='./result', metavar='PREPATH', dest='PREPATH')
    
    args = parser.parse_args()
    
    cancer_data = load_labeled_data(args.data_dir, args.label_list, args.patient_type)
    print('------------cancer_data-----------')
    print(cancer_data)
