import os
import pandas as pd
from IPython.display import HTML

def describe_genes(score_df, ranking_df, gene_descrip):
    prePath = args.PREPATH
    current_path = os.getcwd()    
    
    try: 
        os.makedirs(f'{prePath}') 
    except OSError: 
        if not os.path.isdir(f'{prePath}'): 
            raise       
    
    score_df = pd.read_csv(f'{score_df}', index_col='Unnamed: 0')
    ranking_df = pd.read_csv(f'{ranking_df}', index_col='Unnamed: 0') 
    
    mx = score_df.max().argmax()
    gene_list = ranking_df.sort_values(by=ranking_df.columns[0]).iloc[:score_df.iloc[:,mx].argmax()].index
    
    gene_descrip = pd.read_csv(gene_descrip, sep='\t')
    gene_descrip = gene_descrip[['GeneID', 'Symbol', 'type_of_gene', 'map_location', 'description']]
    
    df = pd.DataFrame(data=None, columns=gene_descrip.columns)
    for gene in gene_list:
        gsymbol, gid = gene.split('|')[0], int(gene.split('|')[1])
        if sum(df['GeneID'] == gid) == 1:
            df = pd.concat([df,gene_descrip.loc[df['GeneID'] == gid]], ignore_index=True)
        elif sum(df['Symbol'] == gsymbol) == 1:
            df = pd.concat([df,gene_descrip.loc[df['Symbol'] == gsymbol]], ignore_index=True)
        else:
            df = df.append({'GeneID': gid, 'Symbol': gsymbol, 'type_of_gene': None, 'map_location': None, 'description': None}, ignore_index=True)

    df['links']= 'https://www.genecards.org/cgi-bin/carddisp.pl?gene='+df['GeneID'].astype(str)
    data = HTML(df.to_html(render_links=True, escape=False))
    
    os.chdir(f'{prePath}')
    df.to_csv('gene_description.csv')
    os.chdir(f'{current_path}')
    
    return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Gene Marker Detection')
    parser.add_argument('--score_df', type=str, help='Score dataframe', required=True)
    parser.add_argument('--ranking_df', type=str, help='Ranking data', required=True)
    parser.add_argument('--gene_descrip', type=str, help='Path where the gene description metadata exists', default='./Metadata/Homo_sapiens.gene_info', required=True)
    parser.add_argument('-prePath',help='path of output file', default='./result',metavar='PREPATH',dest='PREPATH')
    
    args = parser.parse_args()
    
    gene_description = describe_genes(args.score_df, args.ranking_df, args.gene_descrip)
    print('------------gene description-----------')
    print(gene_description)
