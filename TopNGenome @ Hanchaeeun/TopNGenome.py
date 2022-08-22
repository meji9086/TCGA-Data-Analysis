get_ipython().system('pip install matplotlib')
get_ipython().system('pip install seaborn')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def TopN_Genome(df, model, N):
    # 모델 별 중요도 정렬 기준
    sort_dic = {"RF" : False,
                "LGBM" : False,
                "XGB" : False,
                "CAT" : False,
                "Ada" : False
               }
    # 선택한 모델의 N개의 중요도
    data = df.loc[:, model]
    top_df = data.sort_values(ascending=sort_dic[model])[:N]
    # 상위 중요도 유전자 시각화
    fig, ax = plt.subplots(figsize=(12,6))
    ax.set_title(f"{model} Top {N} Biomarker Importances", pad=10, fontsize=18)
    colors = sns.color_palette('Blues_r', len(top_df))
    bar = plt.barh(top_df.index, top_df, color=colors)
    ax.set_yticklabels(top_df.index, fontsize=10)
    for idx, b in enumerate(bar):
        y = b.get_width()
        ax.text(y+0.1, idx, round(y,5), va='bottom', fontsize=10)
    
    plt.show()
    return top_df




