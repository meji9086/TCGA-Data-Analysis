#!pip install requests

import requests
import os
import shutil

def Download_data(data):
    os.mkdir(genome_path)    
    for i in data:
        main_URL = "http://gdac.broadinstitute.org/runs/stddata__2016_01_28/data/{0}/20160128/gdac.broadinstitute.org_LUAD.Merge_rnaseqv2__illuminahiseq_rnaseqv2__unc_edu__Level_3__RSEM_genes_normalized__data.Level_3.2016012800.0.0.tar.gz".format(i)
        sub_URL = "{0}.rnaseqv2__illuminahiseq_rnaseqv2__unc_edu__Level_3__RSEM_genes_normalized__data.data.txt".format(i)
        file = requests.get(main_URL, stream = True)

        with open(sub_URL,"wb") as txt:
            for chunk in file.iter_content(chunk_size=1024):
                if chunk:
                    txt.write(chunk)
                    
        #change path            
        shutil.move('/mnt/workspace/MyFiles/공모전/' + sub_URL, genome_path)             
