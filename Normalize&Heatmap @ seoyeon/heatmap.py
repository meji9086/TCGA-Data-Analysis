get_ipython().system('pip install pandas')
get_ipython().system('pip install sklearn')

import seaborn as sns
import matplotlib.pyplot as plt 

def plot_heatmap(df, gene_list):
    
    cancer_list = list(df['Target'])
    # cancer index
    ind = sorted(np.unique(cancer_list, return_index=True)[1])
    names = [cancer_list[i] for i in sorted(ind)]
    
    # cancer index plot
    pal = sns.husl_palette(len(names), s=.45)
    lut = dict(zip(map(str, np.unique(cancer_list)),pal))
    cancer_colors = np.asarray(pd.Series(cancer_list).map(lut))
    
    # clustermap
    data = df.loc[:,gene_list].transpose()
    cmap = sns.clustermap(data, 
                  cmap = 'RdBu_r',
                  cbar_kws = {"shrink": .8},
                  center = 0.0,
                  vmin = -2,
                  vmax = 2,
                  linewidths=0,
                  xticklabels=False,
                  cbar_pos=(1, .015, .04, .75),
                  row_cluster=True, 
                  col_cluster=False,
                  col_colors= cancer_colors
                )
    
    ind.append(len(cancer_list))
    cmap.ax_heatmap.set_xticks([((ind[i]+ind[i+1])/2) for i in range(len(ind)-1)])
    cmap.ax_heatmap.set_xticklabels(names)
    cmap.ax_heatmap.set_xlabel(' / '.join([i for i in names]), fontsize = 15, labelpad = 25)
    cmap.ax_heatmap.set_title('cancer biomaker heatmap', y = 1.1, fontsize = 25)
