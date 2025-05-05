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

Licensing
--
Copyright (C) 2025 Ying Guo

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3 of the License.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.
