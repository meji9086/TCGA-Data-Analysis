import gzip
import requests
import os
import ssl
import shutil
import time
import tarfile
import pandas as pd


def download_metadata():
    prePath = args.PREPATH
    current_path = os.getcwd()    
    
    try: 
        os.makedirs(f'{prePath}') 
    except OSError: 
        if not os.path.isdir(f'{prePath}'): 
            raise   
            
    csv_file = {
        'cancer_link.csv' : 'https://drive.google.com/uc?export=download&id=1QD2JEkepX6_W1TeQ5sY42R6flhW6833X',
        'BRCApatients_type.csv' : 'https://drive.google.com/uc?export=download&id=1CmjvbF_i6oYzQFWAnwLZORd4tnnW3GvK',
        'Homo_sapiens.gene_info' : 'https://drive.google.com/uc?export=download&id=1IPQq7lBqbpzz2GBNXT2sDh0Hdn4iL8fN'
    }
    names = [file for file in csv_file]
    with open(f'{prePath}{names[0]}', "wb") as file:
        sess = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=100, pool_maxsize=100)
        sess.mount('http://', adapter)
        res = sess.get(csv_file[names[0]])
        file.write(res.content)
    time.sleep(3)
    
    with open(f'{prePath}{names[1]}', "wb") as file:
        sess = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=100, pool_maxsize=100)
        sess.mount('http://', adapter)
        res = sess.get(csv_file[names[1]])
        file.write(res.content)
    time.sleep(3)
    
    with open(f'{prePath}{names[2]}', "wb") as file:
        sess = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=100, pool_maxsize=100)
        sess.mount('http://', adapter)
        res = sess.get(csv_file[names[2]])
        file.write(res.content)    
    print(f'Metadata is installed')
        
        

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Gene Marker Detection')
    parser.add_argument('-prePath', help='path of output file', default='./Metadata/', metavar='PREPATH', dest='PREPATH')
    
    args = parser.parse_args()
    
    download_metadata()