# 💻 TCGA Genome Data Analysis project 💻

본 프로젝트는 Dataon의 2022 연구 데이터 분석활용 경진대회에 참여하며 진행했습니다.

 대회 URL : http://dataon-con.kr/pages/about_new.php
 
 주제 - Gexp : Genemarker Expert 머신러닝 기반 멀티 클래스 분석 바이오 마커 탐지 소프트웨어 

## 👩‍👩‍👧‍👧 Team Info.
|이름|역할|
|:------:|:---:|
|<span style="color:blue">[김예지](https://github.com/meji9086)</span>|Measurement of Ranking and Feature Importance Using Modeling|
|<span style="color:blue">[한채은](https://github.com/Hanchaeeun)</span>|Measure and Compare Accuracy Using Modeling and Visualization|
|<span style="color:blue">[이선우](https://github.com/susan8653)</span>|Data download and extract file Using Web Crawling|
|<span style="color:blue">[강서연](https://github.com/Kangseoyeon512)</span>|Data Visualization Using Heatmap and clustering|

## 🏆 Awarding
🎉 우수상 수상 🎉 

![공모전 수상](https://user-images.githubusercontent.com/72390138/193196125-c1d9000c-8478-4682-a001-3f8ba9b3f916.jpg)


## 📋 Pipeline
### Scripts
```
gexp        
├── download_cancer.py     
├── load_labeled_data.py     
├── biomarker_rank.py       
├── plot_stepwise_accuracy.py     
├── describe_genes.py
├── normalize.py        
├── plot_heatmap.py        
└──            
```


### download_cancer.py
```
Download cancer data (mRNAseq) from the firebrowse site(http://firebrowse.org/)

Optional Argument
 --cancer_list
 --data_source
```

Example            
![image](https://user-images.githubusercontent.com/72390138/193192743-18743150-b79d-4e2b-b596-4be2f0c7353e.png)


### load_labeled_data.py
```
Create a Target variable as part of the preprocessing process

Optional Argument
 --data_dir
 --label_list
 --patient_type
```

Example        
<img src="https://user-images.githubusercontent.com/72390138/193194937-e3b7265a-dab0-407c-bf9d-f2e3640ca11f.png"  width="800" height="350"/>


### biomarker_rank.py
```
Measure and rank feature importance by model(RandomForest, EXtraTrees, XGBoost, AdaBoost, DecisionTree)

Optional Argument
 --cancer_df
 --models
```

Example           
<img src="https://user-images.githubusercontent.com/72390138/193202486-87fa3ae6-1e3d-4049-b00f-c3243f1633ca.png"  width="400" height="400"/>


### plot_stepwise_accuracy.py
```
Visualization of accuracy by step and model(RandomForest, MLP)

Optional Argument
 --cancer_df    
 --ranking_df     
 --model     
 --step_num    
 --metric     
 --multi_class    
```

Example              
<img src="https://user-images.githubusercontent.com/72390138/193201351-f6a51d04-3f0c-4fc1-84ca-81ca22f5055a.png"  width="900" height="500"/>


### describe_genes.py      
```
As a result of performance evaluation, gene information can be viewed as many as the number of genes in the high-performance model

Optional Argument          
 --score_df       
 --ranking_df        
 --gene_descrip       
```

Example             
<img src="https://user-images.githubusercontent.com/72390138/193201778-e59e9ed5-2a52-4dd9-b4c7-1e3af6d64f08.png"  width="800" height="400"/>


### plot_heatmap.py
```
Visualization with clustermap as normalized data for top N biomarker genes

Optional Arguement
 --cancer_df       
 --ranking_df    
 --topN       
 --vmin      
 --vmax      
```

Example            
<img src="https://user-images.githubusercontent.com/72390138/193202285-8e3ae4a2-4477-4901-b352-f9bbc4c3eb03.png"  width="700" height="600"/>
