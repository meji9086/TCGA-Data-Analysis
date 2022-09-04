import numpy as np

def normalize(df, methods):
    normalize_df = df.copy()
    
    for method in methods:
        if method == 'log1p':
            for col in df.columns:
                normalize_df.loc[:,col] = np.log1p(df.loc[:,col])
        
        if method == 'z_score':
            for col in normalize_df.columns:
                m, s = normalize_df.loc[:,col].mean(), normalize_df.loc[:,col].std()
                if s == 0.0:
                    normalize_df.loc[:,col] = 0.0
                else:
                    normalize_df.loc[:,col] = (normalize_df.loc[:,col] - m)/s  
    return normalize_df
