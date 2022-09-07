get_ipython().system('pip install pandas')
get_ipython().system('pip install sklearn')

import seaborn as sns
import matplotlib.pyplot as plt 

def plot_heatmap(df, gene_list):
    cancer_list1 = list(df['Target'])
    cancer_list2 = list(df['Target'].unique())
    ind = sorted(np.unique(cancer_list1, return_index=True)[1])
    names = [cancer_list1[index] for index in sorted(ind)]
    df = df.loc[:,gene_list].transpose()
    dmin, dmax = df.min(), df.max()
    #row_colors = ["b" if  else "r"  for i in range (0,50)]
    plt.figure()
    c1 = sns.clustermap(df, 
                  cmap = 'RdBu_r',
                  cbar_kws = {"shrink": .8},
                  center = 0.0,
                  vmin = -2,
                  vmax = 2,
                  linewidths=0,
                  xticklabels=False,
                  cbar_pos=(1, .015, .04, .75),
                  #row_colors = row_colors
                )
    ind.append(len(cancer_list1))
    c1.ax_heatmap.set_xticks([((ind[i]+ind[i+1])/2) for i in range(len(ind)-1)])
    c1.ax_heatmap.set_xticklabels(cancer_list2)
    c1.ax_heatmap.set_xlabel(' / '.join([i for i in cancer_list2]), fontsize = 13)
    c1.ax_heatmap.set_ylabel('')
    c1.ax_heatmap.set_title('cancer biomaker heatmap', y = 1.2, fontsize = 20)  