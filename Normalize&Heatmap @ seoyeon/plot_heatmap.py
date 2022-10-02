import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_heatmap(cancer_df, ranking_df, topN, vmin, vmax):
    prePath = args.PREPATH
    current_path = os.getcwd()  
    
    try: 
        os.makedirs(f'{prePath}') 
    except OSError: 
        if not os.path.isdir(f'{prePath}'): 
            raise       
    
    cancer_df = pd.read_csv(f'{cancer_df}', index_col='Unnamed: 0', skiprows=[1])
    normalize_df = cancer_df.copy()
    ranking_df = pd.read_csv(f'{ranking_df}', index_col='Unnamed: 0')
    
    normalize_df.drop(columns = 'Target', inplace=True)
    for col in normalize_df.columns:
        normalize_df.loc[:,col] = np.log1p(normalize_df.loc[:,col])

        m, s = normalize_df.loc[:,col].mean(), normalize_df.loc[:,col].std()
        if s == 0.0:
            normalize_df.loc[:,col] = 0.0
        else:
            normalize_df.loc[:,col] = (normalize_df.loc[:,col] - m)/s

    normalize_df = pd.concat([normalize_df, cancer_df[['Target']]], axis=1)

    cancer_list = list(normalize_df['Target'])
    ind = sorted(np.unique(cancer_list, return_index=True)[1])
    names = [cancer_list[i] for i in sorted(ind)]

    pal = sns.husl_palette(len(names), s=.45)
    lut = dict(zip(map(str, np.unique(cancer_list)),pal))
    cancer_colors = np.asarray(pd.Series(cancer_list).map(lut))

    gene_list = ranking_df.sort_values(by=ranking_df.columns[0]).iloc[:topN].index
    hmap_data = normalize_df.loc[:,gene_list].transpose()
    cmap = sns.clustermap(hmap_data, 
                  cmap = 'RdBu_r',
                  cbar_kws = {"shrink": .8},
                  center = 0.0,
                  vmin = vmin,
                  vmax = vmax,
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

    os.chdir(f'{prePath}')
    plt.savefig('heatmap.jpeg')
    os.chdir(f'{current_path}')
        
    
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Gene Marker Detection')
    parser.add_argument('--cancer_df', type=str, help='Cancer data', required=True)
    parser.add_argument('--ranking_df', type=str, help='Ranking data', required=True)
    parser.add_argument('--topN', type=int, help='Number of top N genes', default=10)
    parser.add_argument('--vmin', type=int, help='Minimum value in heatmap', default=-2)
    parser.add_argument('--vmax', type=int, help='Maximum value in heatmap', default=2)
    parser.add_argument('-prePath',help='path of output file', default='./result',metavar='PREPATH',dest='PREPATH')
    
    
    args = parser.parse_args()
    
    plot_heatmap(args.cancer_df, args.ranking_df, args.topN, args.vmin, args.vmax)
    print('The heatmap picture has been saved.')

