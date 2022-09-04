get_ipython().system('pip install pandas')
get_ipython().system('pip install sklearn')

import seaborn as sns
import matplotlib.pyplot as plt 
# from heatmapcluster import heatmapcluster

def plot_heatmap(df):
    df = df.transpose()    
    dmin, dmax = df.min(), df.max()
    fig, ax = plt.subplots(figsize=(10,7))
    sns.heatmap(df, 
                  cmap = 'RdBu_r',
                  cbar_kws = {"shrink": .8},
                  center = 0.0,
                  vmin = -2,
                  vmax = 2,
                  xticklabels=False
                )
    ax.set_xticks([len(df.columns)/4,len(df.columns)*3/4])
    ax.set_xticklabels(['LUAD','LUSC'])
    ax.set_title('LUAD/LUSC')
    ax.set_xlabel('Patients')
    ax.set_ylabel('Genome')
    plt.show()  