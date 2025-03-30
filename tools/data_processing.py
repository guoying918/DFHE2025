import torch
import numpy as np
import math
from . import utils

TEST_LSAMPLE_NUM_PER_CLASS = 5

# get train_loader and test_loader
def extract_patches(data, Row, Column, indices, HalfWidth):
    nBand = data.shape[2]
    nSample = len(indices)
    patches = np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, nSample], dtype=np.float32)
    for i, idx in enumerate(indices):
        patches[:, :, :, i] = data[Row[idx] - HalfWidth: Row[idx] + HalfWidth + 1, Column[idx] - HalfWidth: Column[idx] + HalfWidth + 1, :]
    return patches

def extract_patches_from_largest(largest_patches, large_hw, small_hw):
    """
    Extract smaller patches from the larger patches.
    """
    patch_size = 2 * small_hw + 1
    print('patch_size--------', patch_size)
    center = large_hw  # Center position in the larger patch
    small_patches = largest_patches[center - small_hw:center + small_hw + 1, center - small_hw:center + small_hw + 1,:, :]
    return small_patches

def get_train_test_loader(Data_Band_Scaler, GroundTruth, class_num, shot_num_per_class, HalfWidths=[2, 8]): 
    print(Data_Band_Scaler.shape)
    [nRow, nColumn, nBand] = Data_Band_Scaler.shape

    '''label start'''
    num_class = int(np.max(GroundTruth))
    data_band_scaler = utils.flip(Data_Band_Scaler)
    groundtruth = utils.flip(GroundTruth)
    del Data_Band_Scaler
    del GroundTruth

    G_full = groundtruth
    data_full = data_band_scaler
    [Row, Column] = np.nonzero(G_full)

    del data_band_scaler
    del groundtruth

    nSample = np.size(Row)
    print('number of sample', nSample)

    # Sampling samples
    train = {}
    test = {}
    da_train = {}
    m = int(np.max(G_full))
    nlabeled = TEST_LSAMPLE_NUM_PER_CLASS
    print('labeled number per class:', nlabeled)
    print((200 - nlabeled) / nlabeled + 1)
    print(math.ceil((200 - nlabeled) / nlabeled) + 1)

    for i in range(m):
        indices = [j for j, x in enumerate(Row.ravel().tolist()) if G_full[Row[j], Column[j]] == i + 1]
        np.random.shuffle(indices)
        nb_val = shot_num_per_class
        train[i] = indices[:nb_val]
        da_train[i] = []
        for _ in range(math.ceil((200 - nlabeled) / nlabeled) + 1):
            da_train[i] += indices[:nb_val]
        test[i] = indices[nb_val:]

    train_indices = []
    test_indices = []
    da_train_indices = []
    for i in range(m):
        train_indices += train[i]
        test_indices += test[i]
        da_train_indices += da_train[i]
    np.random.shuffle(test_indices)

    print('the number of train_indices:', len(train_indices))
    print('the number of test_indices:', len(test_indices))
    print('the number of train_indices after data argumentation:', len(da_train_indices))
    print('labeled sample indices:', train_indices)

    nTrain = len(train_indices)
    nTest = len(test_indices)
    da_nTrain = len(da_train_indices)

#################
    imdb = {}
    imdb['data'] = {hw: np.zeros([2 * hw + 1, 2 * hw + 1, nBand, nTrain + nTest], dtype=np.float32) for hw in HalfWidths}
    imdb['Labels'] = np.zeros([nTrain + nTest], dtype=np.int64)
    imdb['set'] = np.zeros([nTrain + nTest], dtype=np.int64)

    RandPerm = train_indices + test_indices
    RandPerm = np.array(RandPerm)
 
    # Extract largest patches once (using the largest HalfWidth)
    max_hw = max(HalfWidths)
    largest_patches = extract_patches(data_full, Row, Column, RandPerm, max_hw)

    # Extract patches for each HalfWidth from the largest patches
    for hw in HalfWidths:
        if hw == max_hw:
            imdb['data'][hw] = largest_patches
        else:
            imdb['data'][hw] = extract_patches_from_largest(largest_patches, max_hw, hw)

    for i, idx in enumerate(RandPerm):
        imdb['Labels'][i] = G_full[Row[idx], Column[idx]].astype(np.int64)
    imdb['Labels'] = imdb['Labels'] - 1
    imdb['set'] = np.hstack((np.ones([nTrain]), 3 * np.ones([nTest]))).astype(np.int64)
    print('Data is OK.')

    train_dataset = utils.matcifar_da(imdb, train=True, d=3, medicinal=0)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=class_num * shot_num_per_class, shuffle=False, num_workers=10) # pin_memory=True
    del train_dataset

    test_dataset = utils.matcifar_da(imdb, train=False, d=3, medicinal=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=10) # batch_size=100
    del test_dataset
    del imdb
##################################################

    # Data Augmentation for target domain for training
    imdb_da_train = {}
    imdb_da_train['data'] = {hw: np.zeros([2 * hw + 1, 2 * hw + 1, nBand, da_nTrain], dtype=np.float32) for hw in HalfWidths}
    imdb_da_train['Labels'] = np.zeros([da_nTrain], dtype=np.int64)
    imdb_da_train['set'] = np.zeros([da_nTrain], dtype=np.int64)

    da_RandPerm = np.array(da_train_indices) # (1800,)
    add_nosize_data = utils.radiation_noise(data_full)

    # Extract largest patches once (using the largest HalfWidth)
    max_hw = max(HalfWidths)
    largest_patches_da = extract_patches(add_nosize_data, Row, Column, da_RandPerm, max_hw)

    # Extract patches for each HalfWidth from the largest patches
    for hw in HalfWidths:
        if hw == max_hw:
            imdb_da_train['data'][hw] = largest_patches_da
        else:
            imdb_da_train['data'][hw] = extract_patches_from_largest(largest_patches_da, max_hw, hw)
    for iSample in range(da_nTrain):
        imdb_da_train['Labels'][iSample] = G_full[Row[da_RandPerm[iSample]], Column[da_RandPerm[iSample]]].astype(np.int64)

    imdb_da_train['Labels'] = imdb_da_train['Labels'] - 1
    imdb_da_train['set'] = np.ones([da_nTrain]).astype(np.int64)
    print('Data Augmentation is OK')
    return train_loader, test_loader, imdb_da_train, G_full, RandPerm, Row, Column, nTrain


def get_target_dataset(Data_Band_Scaler, GroundTruth, class_num, shot_num_per_class):
    train_loader, test_loader, imdb_da_train, G, RandPerm, Row, Column, nTrain = get_train_test_loader(Data_Band_Scaler=Data_Band_Scaler,  GroundTruth=GroundTruth, \
                                                                     class_num=class_num,shot_num_per_class=shot_num_per_class)  # 9 classes and 5 labeled samples per class
    target_da_datas = {hw: np.transpose(imdb_da_train['data'][hw], (3, 2, 0, 1)) for hw in imdb_da_train['data']}
    print({hw: target_da_datas[hw].shape for hw in target_da_datas}) # {2: (1800, 100, 5, 5), 8: (1800, 100, 17, 17)}
    target_da_labels = imdb_da_train['Labels']
    print('target data augmentation label:', target_da_labels)
    
    target_da_metatrain_data = {}
    for hw, target_da_data in target_da_datas.items():
        target_da_train_set = {}
        for class_, path in zip(target_da_labels, target_da_data):
            if class_ not in target_da_train_set:
                target_da_train_set[class_] = []
            target_da_train_set[class_].append(path)
        target_da_metatrain_data[hw] = target_da_train_set
    return train_loader, test_loader, target_da_metatrain_data,G,RandPerm,Row, Column,nTrain
