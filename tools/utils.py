import torch
import torch.utils.data as Torchdata
from torch.utils.data import DataLoader, Dataset
import numpy as np
from scipy import io
import random
from sklearn.metrics import confusion_matrix
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from sklearn.decomposition import PCA
from torch.utils.data.sampler import Sampler
from sklearn import preprocessing
import scipy.io as sio

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import spectral
import cv2
from sklearn.cluster import KMeans

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
def flip(data):
    y_4 = np.zeros_like(data)
    y_1 = y_4
    y_2 = y_4
    first = np.concatenate((y_1, y_2, y_1), axis=1)
    second = np.concatenate((y_4, data, y_4), axis=1)
    third = first
    Data = np.concatenate((first, second, third), axis=0)
    return Data

def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1/25):
    alpha = np.random.uniform(*alpha_range)
    noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
    return alpha * data + beta * noise

def flip_augmentation(data): # arrays tuple 0:(7, 7, 103) 1=(7, 7)
    horizontal = np.random.random() > 0.5 # True
    vertical = np.random.random() > 0.5 # False
    if horizontal:
        data = np.fliplr(data)
    if vertical:
        data = np.flipud(data)
    return data

import torch.utils.data as data


from torch.utils.data import Dataset

class matcifar_da(Dataset):
    def __init__(self, imdb, train, d, medicinal):
        self.train = train
        self.imdb = imdb
        self.d = d

        self.x1 = np.argwhere(self.imdb['set'] == 1).flatten()
        self.x2 = np.argwhere(self.imdb['set'] == 3).flatten()

        if medicinal == 1:
            self.train_data = {hw: self.imdb['data'][hw][:, :, :, self.x1] for hw in self.imdb['data']}
            self.train_labels = self.imdb['Labels'][self.x1]
            self.test_data = {hw: self.imdb['data'][hw][:, :, :, self.x2] for hw in self.imdb['data']}
            self.test_labels = self.imdb['Labels'][self.x2]
        else:
            self.train_data = {hw: self.imdb['data'][hw][:, :, :, self.x1] for hw in self.imdb['data']}
            self.train_labels = self.imdb['Labels'][self.x1]
            self.test_data = {hw: self.imdb['data'][hw][:, :, :, self.x2] for hw in self.imdb['data']}
            self.test_labels = self.imdb['Labels'][self.x2]
            for hw in self.imdb['data']:
                if self.d == 3:
                    self.train_data[hw] = self.train_data[hw].transpose((3, 2, 0, 1))
                    self.test_data[hw] = self.test_data[hw].transpose((3, 2, 0, 1))
                else:
                    self.train_data[hw] = self.train_data[hw].transpose((3, 0, 2, 1))
                    self.test_data[hw] = self.test_data[hw].transpose((3, 0, 2, 1))

    def __getitem__(self, index):
        if self.train:
            img = {hw: self.train_data[hw][index] for hw in self.train_data}
            target = self.train_labels[index]
        else:
            img = {hw: self.test_data[hw][index] for hw in self.test_data}
            target = self.test_labels[index]
        return img, target
    
    def __len__(self):
        return len(self.train_labels) if self.train else len(self.test_labels)

class matcifar(data.Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    def __init__(self, imdb, train, d, medicinal):

        self.train = train  # training set or test set
        self.imdb = imdb
        self.d = d

        self.x1 = np.argwhere(self.imdb['set'] == 1)
        self.x2 = np.argwhere(self.imdb['set'] == 3)
        self.x1 = self.x1.flatten()
        self.x2 = self.x2.flatten()
        if medicinal == 1:
            self.train_data = self.imdb['data'][self.x1, :, :, :]
            self.train_labels = self.imdb['Labels'][self.x1]
            self.test_data = self.imdb['data'][self.x2, :, :, :]
            self.test_labels = self.imdb['Labels'][self.x2]

        else:
            self.train_data = self.imdb['data'][:, :, :, self.x1]
            self.train_labels = self.imdb['Labels'][self.x1]
            self.test_data = self.imdb['data'][:, :, :, self.x2]
            self.test_labels = self.imdb['Labels'][self.x2]
            if self.d == 3:
                self.train_data = self.train_data.transpose((3, 2, 0, 1))  ##(17, 17, 200, 10249)
                self.test_data = self.test_data.transpose((3, 2, 0, 1))
            else:
                self.train_data = self.train_data.transpose((3, 0, 2, 1))
                self.test_data = self.test_data.transpose((3, 0, 2, 1))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:

            img, target = self.train_data[index], self.train_labels[index]
        else:

            img, target = self.test_data[index], self.test_labels[index]

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

def sanity_check(all_set):
    nclass = 0
    nsamples = 0
    all_good = {}
    for class_ in all_set:
        if len(all_set[class_]) >= 200:
            all_good[class_] = all_set[class_][:200]
            nclass += 1
            nsamples += len(all_good[class_])
    print('the number of class:', nclass)
    print('the number of sample:', nsamples)
    return all_good

def applyPCA(X, numComponents=75):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents,whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0],X.shape[1], numComponents))
    return newX, pca

def data_load_TIFHDR_PCA(folder, numComponents):
    image_data = spectral.open_image(folder + '2018_IEEE_GRSS_DFC_HSI_TR.HDR')
    img = image_data.load()[:,:,:-2]
    label_data = imageio.imread(folder + '2018_IEEE_GRSS_DFC_GT_TR.tif')
    
    label_values = ["Unclassified","Healthy grass","Stressed grass","Artificial turf",
                        "Evergreen trees","Deciduous trees","Bare earth","Water","Residential buildings",
                        "Non-residential buildings","Roads","Sidewalks","Crosswalks","Major thoroughfares",
                        "Highways","Railways","Paved parking lots","Unpaved parking lots","Cars",
                        "Trains","Stadium seats"]
    rgb_bands = (59, 40, 23)
    
    data_all, pca = applyPCA(img, numComponents=numComponents)
    label_data = cv2.resize(label_data, dsize=(image_data.shape[1],image_data.shape[0]), interpolation=cv2.INTER_NEAREST)
    GroundTruth = label_data

    [nRow, nColumn, nBand] = data_all.shape
    print('Houston2018', nRow, nColumn, nBand)

    data = data_all.reshape(np.prod(data_all.shape[:2]), np.prod(data_all.shape[2:])) 
    data_scaler = preprocessing.scale(data)   
    Data_Band_Scaler = data_scaler.reshape(data_all.shape[0], data_all.shape[1],data_all.shape[2])

    return Data_Band_Scaler, GroundTruth, None

def load_data(dataset_name, folder='./Datasets/test_ocbs/'):
    if dataset_name == 'PaviaU':
        image_data = sio.loadmat(folder + 'PaviaU_data.mat')['A_ocbs'] 
        GroundTruth = sio.loadmat(folder + 'PaviaU_gt.mat')['paviaU_gt']
        label_values = ['Undefined', 'Asphalt', 'Meadows', 'Gravel', 'Trees',
                        'Painted metal sheets', 'Bare Soil', 'Bitumen',
                        'Self-Blocking Bricks', 'Shadows']
        rgb_bands = (55, 41, 12)
    elif dataset_name == 'Salinas':
        image_data = sio.loadmat(folder + 'Salinas_corrected_data.mat')['A_ocbs']
        GroundTruth = sio.loadmat(folder + 'Salinas_gt.mat')['salinas_gt']
        label_values = ['Undefined', 'Brocoli_green_weeds_1', 'Brocoli_green_weeds_2', 'Fallow', 
                        'Fallow_rough_plow', 'Fallow_smooth', 'Stubble', 'Celery', 'Grapes_untrained', 
                        'Soil_vinyard_develop', 'Corn_senesced_green_weeds', 'Lettuce_romaine_4wk', 
                        'Lettuce_romaine_5wk', 'Lettuce_romaine_6wk', 'Lettuce_romaine_7wk', 
                        'Vinyard_untrained', 'Vinyard_vertical_trellis']
        rgb_bands = (43, 21, 11)
    elif dataset_name == 'IndianP':
        image_data = sio.loadmat(folder + 'Indian_pines_corrected_data.mat')['A_ocbs']
        GroundTruth = sio.loadmat(folder + 'Indian_pines_gt.mat')['indian_pines_gt']
        label_values = ['Undefined', 'Brocoli_green_weeds_1', 'Brocoli_green_weeds_2', 'Fallow', 
                        'Fallow_rough_plow', 'Fallow_smooth', 'Stubble', 'Celery', 'Grapes_untrained', 
                        'Soil_vinyard_develop', 'Corn_senesced_green_weeds', 'Lettuce_romaine_4wk', 
                        'Lettuce_romaine_5wk', 'Lettuce_romaine_6wk', 'Lettuce_romaine_7wk', 
                        'Vinyard_untrained', 'Vinyard_vertical_trellis']
        rgb_bands = (43, 21, 11)
    elif dataset_name == 'SH2HZ':
        data = sio.loadmat('/home/guoying/DM-MRN/Datasets/Shanghai-Hangzhou/DataCube.mat')
        data_cube1 = data['DataCube1']
        data_cube2 = data['DataCube2']
        gt1 = data['gt1']
        gt2 = data['gt2']
        # 合并数据立方体
        image_data = np.concatenate((data_cube1, data_cube2), axis=0) # (2190, 230, 198)

        # 合并标签
        GroundTruth = np.concatenate((gt1, gt2), axis=0)
    [nRow, nColumn, nBand] = image_data.shape
    print(nRow, nColumn, nBand)

    # 标准化处理，均值为1，标准差为1，正态分布
    data = image_data.reshape(np.prod(image_data.shape[:2]), np.prod(image_data.shape[2:]))  # (111104,204)
    data_scaler = preprocessing.scale(data)  # (X-X_mean)/X_std,
    Data_Band_Scaler = data_scaler.reshape(image_data.shape[0], image_data.shape[1],image_data.shape[2])

    return Data_Band_Scaler, GroundTruth

def sanity_check(all_set):
    nclass = 0
    nsamples = 0
    all_good = {}
    for class_ in all_set:
        if len(all_set[class_]) >= 200:
            all_good[class_] = all_set[class_][:200]
            nclass += 1
            nsamples += len(all_good[class_])
    print('the number of class:', nclass)
    print('the number of sample:', nsamples)
    return all_good

class Task(object):
    def __init__(self, data_dict, num_classes, shot_num, query_num):
        self.data_dict = data_dict
        self.num_classes = num_classes
        self.support_num = shot_num
        self.query_num = query_num

        class_folders = sorted(list(next(iter(data_dict.values())).keys()))
        # class_folders = sorted(list(next(iter(data_dict.values()))))
        class_list = random.sample(class_folders, self.num_classes) # 9
        labels = np.array(range(len(class_list)))
        labels = dict(zip(class_list, labels))

        self.support_datas = {scale: [] for scale in data_dict.keys()}
        self.query_datas = {scale: [] for scale in data_dict.keys()}
        self.support_labels = []
        self.query_labels = []
        for c in class_list:
            temp = list(range(len(data_dict[next(iter(data_dict))][c]))) 
            indices = random.sample(temp, shot_num + query_num)
            
            for scale, data in data_dict.items():
                temp = data[c]
                selected_samples = [temp[idx] for idx in indices]
                self.support_datas[scale] += selected_samples[:shot_num]
                self.query_datas[scale] += selected_samples[shot_num:shot_num + query_num]
            self.support_labels += [labels[c] for _ in range(shot_num)]
            self.query_labels += [labels[c] for _ in range(query_num)]
        
class FewShotDataset(Dataset):
    def __init__(self, task, split='train'):
        self.task = task
        self.split = split
        self.image_datas = self.task.support_datas if self.split == 'train' else self.task.query_datas
        self.labels = self.task.support_labels if self.split == 'train' else self.task.query_labels

    def __len__(self):
        return len(next(iter(self.image_datas.values())))

    def __getitem__(self, idx):
        raise NotImplementedError("This is an abstract class. Subclass this class for your particular dataset.")

class HBKC_dataset(FewShotDataset):
    def __init__(self, *args, **kwargs):
        super(HBKC_dataset, self).__init__(*args, **kwargs)
        
    def __getitem__(self, idx):
        images = {scale: torch.tensor(data[idx]) for scale, data in self.image_datas.items()}
        label = torch.tensor(self.labels[idx])
        return images, label

# Sampler
class ClassBalancedSampler(Sampler):
    ''' Samples 'num_inst' examples each from 'num_cl' pool of examples of size 'num_per_class' '''
    def __init__(self, num_per_class, num_cl, num_inst, shuffle=False):
        self.num_per_class = num_per_class
        self.num_cl = num_cl
        self.num_inst = num_inst
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            batch = [[i + j * self.num_inst for i in torch.randperm(self.num_inst)[:self.num_per_class]] for j in range(self.num_cl)]
        else:
            batch = [[i + j * self.num_inst for i in range(self.num_inst)[:self.num_per_class]] for j in range(self.num_cl)]
        batch = [item for sublist in batch for item in sublist]
        
        if self.shuffle:
            random.shuffle(batch)
        return iter(batch)

    def __len__(self):
        return 1
def collate_fn(batch):
    images_batch, labels_batch = zip(*batch)
    scales = list(images_batch[0].keys())
    images_batch_collated = {scale: torch.stack([images[scale] for images in images_batch]) for scale in scales}
    labels_batch_collated = torch.tensor(labels_batch)
    
    return images_batch_collated, labels_batch_collated

# dataloader
def get_HBKC_data_loader(task, num_per_class, split='train', shuffle=False, ba_type="base"):
    dataset = HBKC_dataset(task, split=split)

    if split == 'train':
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.support_num, shuffle=shuffle)  # support set
    else:
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.query_num, shuffle=shuffle)  # query set
       
    loader = DataLoader(dataset, batch_size=num_per_class * task.num_classes, sampler=sampler, collate_fn=collate_fn)
    
    return loader

def classification_map(map, groundTruth, dpi, savePath):

    fig = plt.figure(frameon=False)
    fig.set_size_inches(groundTruth.shape[1]*2.0/dpi, groundTruth.shape[0]*2.0/dpi)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)

    ax.imshow(map)
    fig.savefig(savePath, dpi = dpi)

    return 0

def split_first_dim_linear(x, first_two_dims):
    """
    Undo the stacking operation
    """
    x_shape = x.size()
    new_shape = first_two_dims
    if len(x_shape) > 1:
        new_shape += [x_shape[-1]]
    return x.view(new_shape)

def mean_pooling(x):
    return torch.mean(x, dim=0, keepdim=True)
