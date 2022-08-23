# !pip install pandas
# !pip install seaborn
# !pip install matplotlib

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from DataLoader import Load_Labeling

def heat_zscore(df, cancer_list, genome_list):
    d1 = df[genome_list]  #최종 data
    z_df = d1.copy()
    
    for idx, c in enumerate(d1.columns):
        #Z-Score
        m, s = d1.loc[:,c].mean(), d1.loc[:,c].std()
        if s == 0.0:
            z_df.loc[:,c] = 0.0
        else:
            z_df.loc[:,c] = (d1.loc[:,c] - m)/s    
    #heatmap
    z_df = z_df.transpose()    
    dmin, dmax = z_df.min(), z_df.max()
    fig, ax = plt.subplots(figsize=(10,7))
    sns.heatmap(z_df, 
                  cmap = 'RdBu_r',
                  cbar_kws = {"shrink": .8},
                  center = 0.0,
                  vmin = -2,
                  vmax = 2
                )

    ax.set_title('LUAD/LUSC')
    ax.set_xlabel('Patients')
    ax.set_ylabel('Genome')
    plt.show()  
    
    return z_df


def heat_minmax(df, cancer_list, genome_list):
    d2 = df[genome_list]  #최종 data
    mm_df = d2.copy()
    
    for idx, c in enumerate(d2.columns):
        #min max scaling
        min, max = d2.loc[:,c].min(), d2.loc[:,c].max()
        if (max - min) == 0.0:
            mm_df.loc[:,c] = 0.0
        else:
            mm_df.loc[:,c] = (d2.loc[:,c] - min)/(max - min)
    #heatmap
    mm_df = mm_df.transpose()    
    dmin, dmax = mm_df.min(), mm_df.max()
    fig, ax = plt.subplots(figsize=(10,7))
    sns.heatmap(mm_df, 
                  cmap = 'RdBu_r',
                  cbar_kws = {"shrink": .8},
                  center = 0.5,
                  vmin = 0.0,
                  vmax = 1.0
                )

    ax.set_title('LUAD/LUSC')
    ax.set_xlabel('Patients')
    ax.set_ylabel('Genome')
    plt.show()  
    
    return mm_df


def heat_maxabs(df, cancer_list, genome_list):
    d3 = df[genome_list]  #최종 data
    ma_df = d3.copy()
    
    for idx, c in enumerate(d3.columns):
        #max_abs
        m_abs = d3.loc[:,c].abs().max()
        if m_abs == 0.0:
            ma_df.loc[:,c] = 0.0
        else:
            ma_df.loc[:,c] = d3.loc[:,c]/m_abs
    #heatmap
    ma_df = ma_df.transpose()    
    dmin, dmax = ma_df.min(), ma_df.max()
    fig, ax = plt.subplots(figsize=(10,7))
    sns.heatmap(ma_df, 
                  cmap = 'RdBu_r',
                  cbar_kws = {"shrink": .8},
                  center = 0.0,
                  vmin = -1,
                  vmax = 1
                )

    ax.set_title('LUAD/LUSC')
    ax.set_xlabel('Patients')
    ax.set_ylabel('Genome')
    plt.show()  
    
    return ma_df
