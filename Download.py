#get_ipython().system('pip install requests')

import requests
import os
import shutil
import pandas as pd
import tarfile

def download_data(cancer_list, data_source, data_dir=None):
    #path setting
    current_path = os.getcwd()
    if data_dir is None:
        path = current_path
    else:
        path = f'{current_path}/{data_dir}'
    # download path file 
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
        #Uncompressing        
        fname = f'./{local_filename}'  # 압축 파일을 지정해주고   
        ap = tarfile.open(fname)      # 열어줍니다. 
        ap.extractall(path)
        ap.close()      
        f = local_filename.rstrip('.tar.gz')
        entries = os.listdir(f'{path}/{f}')
        for entry in entries:
            if cancer in entry:
                shutil.move(f'{path}/{f}/{entry}', path)
                shutil.rmtree(f'{path}/{f}')