import gzip
import requests
import os
import shutil
import time
import tarfile
import pandas as pd


def download_cancer(cancer_list, data_source):
    prePath = args.PREPATH
    current_path = os.getcwd()    
    
    try: 
        os.makedirs(f'{prePath}') 
    except OSError: 
        if not os.path.isdir(f'{prePath}'): 
            raise       
    
            
    down_folder = [f.split(".")[0] for f in os.listdir(prePath) if f != '.ipynb_checkpoints']
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
                    ap.extractall(prePath)
                    ap.close()

                    folder = local_filename.rstrip('.tar.gz')
                    entries = os.listdir(f'{prePath}{folder}')
                    for entry in entries:
                        if cancer in entry:
                            shutil.move(f'{prePath}{folder}/{entry}', prePath)
                            shutil.rmtree(f'{prePath}{folder}')
                    os.remove(f'{fname}') 
                else:
                     break
                            
            except:
                time.sleep(1)    
            print(f'The {cancer} is downloaded successfully')
            
            
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Gene Marker Detection')
    parser.add_argument('--cancer_list', nargs='+', help='Cancer name', required=True)
    parser.add_argument('--data_source', type=str, help='Path where the gene description cancer_link.csv exists', default='./Metadata/cancer_link.csv')
    parser.add_argument('-prePath', help='path of output file', default='./Data/',metavar='PREPATH',dest='PREPATH')
    
    
    args = parser.parse_args()
    
    download_cancer(args.cancer_list, args.data_source)
