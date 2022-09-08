import numpy as np
import pandas as pd

def normalize(df, methods, exclude):
    normalize_df = df.copy()
    normalize_df.drop(columns = exclude, inplace=True)

    for method in methods:
        if method == 'log1p':
            for col in normalize_df.columns:
                normalize_df.loc[:,col] = np.log1p(normalize_df.loc[:,col])
        
        if method == 'z_score':
            for col in normalize_df.columns:
                m, s = normalize_df.loc[:,col].mean(), normalize_df.loc[:,col].std()
                if s == 0.0:
                    normalize_df.loc[:,col] = 0.0
                else:
                    normalize_df.loc[:,col] = (normalize_df.loc[:,col] - m)/s  
    normalize_df = pd.concat([normalize_df, df[[exclude]]], axis=1)
    return normalize_df
