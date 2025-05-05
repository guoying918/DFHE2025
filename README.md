# DFHE2025

# The code for our research on few-shot hyperspectral image classification method.



## Datasets

```
├── Patch5_TRIAN_META_DATA_imdb_ocbs.pickle
├── Patch17_TRIAN_META_DATA_imdb_ocbs.pickle
├── test_ocbs
│   ├── PaviaU_data.mat
│   ├── PaviaU_gt.mat
└── train_ocbs
    ├── Botswana_data.mat
    ├── Botswana_gt.mat
    ├── Chikusei_data.mat
    ├── Chikusei_gt.mat
    ├── KSC_data.mat
    └── KSC_gt.mat
```
1) Please prepare the training and test data as operated in the paper. The used OCBS band selection method is referred to https://github.com/tanmlh.
2) Run "trainMetaDataProcess.py" to generate the meta-training data "Patch5_TRIAN_META_DATA_imdb_ocbs.pickle" and "Patch17_TRIAN_META_DATA_imdb_ocbs.pickle". 
3) Run "python DFHE.py".

