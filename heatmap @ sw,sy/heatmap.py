# !pip install pandas
# !pip install seaborn
# !pip install matplotlib

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from DataLoader import Load_Labeling

def heat_zscore(df, cancer_list, genome_list):
    d1 = pd.DataFrame()  #최종 data
    for idx, c in enumerate(cancer_list):
        data_select = df[df['LUSC']==idx]
        Topdf = data_select[genome_list]
        #Topdf
    
        #Z-Score
        z_score = (Topdf - Topdf.mean())/Topdf.std()
        z_score = z_score.set_axis([c]*len(z_score), axis=0)
        z_score = z_score.transpose()
        d1 = pd.concat([d1, z_score], axis=1)
    #heatmap
    fig, ax = plt.subplots(figsize=(10,7))

    sns.heatmap(d1,
                  cmap = 'RdBu_r',
                  cbar_kws = {"shrink": .8}
                )

    ax.set_title('LUAD/LUSC')
    ax.set_xlabel('Patients')
    ax.set_ylabel('Genome')

    plt.show()    
    return d1


def heat_minmax(df, cancer_list, genome_list):
    
    d2 = pd.DataFrame()  #최종 data
    for idx, c in enumerate(cancer_list):
        data_select = df[df['LUSC']==idx]
        Topdf = data_select[genome_list]
        #Topdf
    
        #min-max scaling
        min_max = (Topdf - Topdf.min())/(Topdf.max() - Topdf.min())
        min_max = min_max.set_axis([c]*len(min_max), axis=0)
        min_max = min_max.transpose()
        d2 = pd.concat([d2, min_max], axis=1)
    #heatmap
    fig, ax = plt.subplots(figsize=(10,7))

    sns.heatmap(d2,
            cmap = 'RdBu_r',
            cbar_kws = {"shrink": .8}
                )

    ax.set_title('LUAD/LUSC')
    ax.set_xlabel('Patients')
    ax.set_ylabel('Genome')

    plt.show()
    return d2


def heat_maxabs(df, cancer_list, genome_list):
    
    d3 = pd.DataFrame()  #최종 data
    for idx, c in enumerate(cancer_list):
        data_select = df[df['LUSC']==idx]
        Topdf = data_select[genome_list]
        #Topdf
    
        #max abs
        max_abs = Topdf/Topdf.abs().max()
        max_abs = max_abs.set_axis([c]*len(max_abs), axis=0)
        max_abs = max_abs.transpose()
        d3 = pd.concat([d3, max_abs], axis=1)
    #heatmap
    fig, ax = plt.subplots(figsize=(10,7))

    sns.heatmap(d3,
            cmap = 'RdBu_r',
            cbar_kws = {"shrink": .8}
                )

    ax.set_title('LUAD/LUSC')
    ax.set_xlabel('Patients')
    ax.set_ylabel('Genome')

    plt.show()
    return d3
