import torch
import torch.nn as nn
import numpy as np
import argparse
import pickle
import time
import datetime
from sklearn import metrics
import os 
from tools.modelStatsRecord import OutputData
from models_tools import *
from tools.data_processing import *
import tools.utils as utils
from Models.Net import *

parser = argparse.ArgumentParser(description="Few Shot Visual Recognition")
parser.add_argument("-dataset","--dataset",type = str, default = 'PaviaU') # PaviaU、Salinas、IndianP、Houston2018
parser.add_argument("-f","--feature_dim",type = int, default = 192)
parser.add_argument("-c","--src_input_dim",type = int, default = 100)
parser.add_argument("-n","--n_dim",type = int, default = 100)
parser.add_argument("-w","--class_num",type = int, default = 9)
parser.add_argument("-s","--shot_num_per_class",type = int, default = 5)
parser.add_argument("-b","--query_num_per_class",type = int, default = 19)
parser.add_argument("-e","--episode",type = int, default= 10000)
parser.add_argument("-l","--learning_rate", type = float, default = 0.001)
parser.add_argument("-g","--gpu", type = str, default = '1')
# target
parser.add_argument("-m","--test_class_num",type=int, default=9)
parser.add_argument("-z","--test_lsample_num_per_class",type=int,default=5, help='1 2 3 4 5')

args = parser.parse_args()
GPU = args.gpu
# Hyper Parameters
DATASET = args.dataset
FEATURE_DIM = args.feature_dim
SRC_INPUT_DIMENSION = args.src_input_dim
N_DIMENSION = args.n_dim
CLASS_NUM = args.class_num
SHOT_NUM_PER_CLASS = args.shot_num_per_class
QUERY_NUM_PER_CLASS = args.query_num_per_class
TEST_CLASS_NUM = args.test_class_num # the number of class
TEST_LSAMPLE_NUM_PER_CLASS = args.test_lsample_num_per_class # the number of labeled samples per class 5 4 3 2 1
EPISODE = args.episode
LEARNING_RATE = args.learning_rate


current_date = datetime.date.today().strftime("%Y%m%d")
RESULT_DIR ='./' + current_date +'_result_multipatch/' + DATASET + '/'
if not os.path.isdir(RESULT_DIR):
    os.makedirs(RESULT_DIR)
CHECKPOINT_PATH = "./checkpoints/"+ DATASET + "/"
if not os.path.isdir(CHECKPOINT_PATH):
    os.makedirs(CHECKPOINT_PATH)
CLASSIFICATIONMAP_PATH = "./classificationMaps/"+ DATASET + "/"
if not os.path.isdir(CLASSIFICATIONMAP_PATH):
    os.makedirs(CLASSIFICATIONMAP_PATH)

utils.same_seeds(0)

def list_files_in_directory(directory):
    items = os.listdir(directory)
    files = [item for item in items if os.path.isfile(os.path.join(directory, item))]
    return files

directory_path = "./Datasets/"

def data_read(files):
    metatrain_data_dict = {}
    for file in files:
        # load source domain data set
        print(file)
        hw = file.split('_')[0]
        with open(os.path.join(directory_path, file), 'rb') as handle:
            source_imdb = pickle.load(handle)
        source_imdb['data']=np.array(source_imdb['data'])
        source_imdb['Labels']=np.array(source_imdb['Labels'],dtype='int')
        source_imdb['set']=np.array(source_imdb['set'],dtype='int')

        # process source domain data set
        data_train = source_imdb['data'] # (86874, 9, 9, 100)
        labels_train = source_imdb['Labels'] # (86874,)
        keys_all_train = sorted(list(set(labels_train)))  # class [0,...,45]
        label_encoder_train = {}  #{0: 0, 1: 1, 2: 2, 3: 3,...,45: 45}
        for i in range(len(keys_all_train)):
            label_encoder_train[keys_all_train[i]] = i

        train_set = {}
        for class_, path in zip(labels_train, data_train):
            if label_encoder_train[class_] not in train_set:
                train_set[label_encoder_train[class_]] = []
            train_set[label_encoder_train[class_]].append(path)
        data = train_set
        del train_set
        del keys_all_train
        del label_encoder_train

        print("Num classes for source domain datasets: " + str(len(data)))
        print(data.keys())
        data = utils.sanity_check(data) # 200 labels samples per class
        print("Num classes of the number of class larger than 200: " + str(len(data))) # 40 classes  8000 samples

        for class_ in data:
            for i in range(len(data[class_])):
                image_transpose = np.transpose(data[class_][i], (2, 0, 1))
                data[class_][i] = image_transpose

        # source few-shot classification data
        metatrain_data = data
        print(len(metatrain_data.keys()), metatrain_data.keys()) # 40 classes
        metatrain_data_dict[hw] = metatrain_data
        del data
    return metatrain_data_dict

sorted_files = ['Patch5_TRIAN_META_DATA_imdb_ocbs.pickle', 'Patch17_TRIAN_META_DATA_imdb_ocbs.pickle']
base_datasets = data_read(sorted_files)

FOLDER = './Datatets/target/'

# loader targer datasets
if DATASET == 'Houston2018':
    Data_Band_Scaler, GroundTruth,_ = utils.data_load_TIFHDR_PCA(FOLDER, numComponents=10)
else:
    Data_Band_Scaler, GroundTruth = utils.load_data(DATASET, FOLDER)

print('Finished load dataset')

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}
def to_cuda(tensor_dict):
    return {scale: tensor.cuda() for scale, tensor in tensor_dict.items()}

crossEntropy = nn.CrossEntropyLoss().cuda()

# run 10 times
nDataSet = 1
acc = np.zeros([nDataSet, 1])
A = np.zeros([nDataSet, TEST_CLASS_NUM])
P = np.zeros([nDataSet, TEST_CLASS_NUM])
k = np.zeros([nDataSet, 1])
training_time = np.zeros([nDataSet, 1])
test_time = np.zeros([nDataSet, 1])
best_predict_all = []
best_acc_all = 0.0
best_G,best_RandPerm,best_Row, best_Column,best_nTrain = None,None,None,None,None
latest_G,latest_RandPerm,latest_Row, latest_Column,latest_nTrain = None,None,None,None,None

seeds = [1330, 1220, 1336, 1337, 1224, 1236, 1226, 1235, 1233, 1229]
for iDataSet in range(nDataSet):
    print('iDataSet--------', iDataSet)
    # load target domain data for training and testing
    np.random.seed(seeds[iDataSet])
    train_loader, test_loader, target_da_metatrain_data,G,RandPerm,Row, Column,nTrain = get_target_dataset(
        Data_Band_Scaler=Data_Band_Scaler, GroundTruth=GroundTruth,class_num=TEST_CLASS_NUM, shot_num_per_class=TEST_LSAMPLE_NUM_PER_CLASS)
    feature_encoder = feature_encode(TEST_CLASS_NUM)
    print(get_parameter_number(feature_encoder))  # {'Total': 1519081, 'Trainable': 1519081}

    feature_encoder.cuda()
    feature_encoder.train()
    # optimizer
    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(), lr=args.learning_rate)

    last_accuracy = 0.0
    best_episdoe = 0
    train_loss = []
    test_acc = []
    total_hit, total_num = 0.0, 0.0
    test_acc_list = []

    torch.cuda.synchronize()
    train_start = time.time()
    EPISODE_1 = 1000
    N_RUNS = 10
    flag = 1
    for i in range(N_RUNS):
        for episode in range(EPISODE_1): 
            # source domain few-shot
            if episode < 800:
                '''Few-shot claification for source domain data set'''
                # get few-shot classification samples
                task = utils.Task(base_datasets, CLASS_NUM, SHOT_NUM_PER_CLASS, QUERY_NUM_PER_CLASS)  # 9, 180, 19
                support_dataloader = utils.get_HBKC_data_loader(task, num_per_class=SHOT_NUM_PER_CLASS, split="train", shuffle=False)
                query_dataloader = utils.get_HBKC_data_loader(task, num_per_class=QUERY_NUM_PER_CLASS, split="test", shuffle=True)

                # sample datas
                supports, support_labels = support_dataloader.__iter__().__next__()  # (5, 100, 9, 9)
                querys, query_labels = query_dataloader.__iter__().__next__()  # (75,100,9,9)

                # calculate features
                supports = to_cuda(supports)
                querys = to_cuda(querys)
                support_proto_features, query_Enhance_features, per_outputs, enhance_loss = feature_encoder(supports, querys, support_labels.cuda(), query_labels.cuda(), domain='source')
                logits = MD_distance_0(support_proto_features, per_outputs, support_labels, query_Enhance_features)
                f_loss = crossEntropy(logits, query_labels.cuda())
                loss = f_loss + enhance_loss
                # Update parameters
                feature_encoder_optim.zero_grad()
                loss.backward()
                feature_encoder_optim.step()
                
                total_hit += torch.sum(torch.argmax(logits, dim=1).cpu() == query_labels).item()
                total_num += query_labels.shape[0]

            if episode >= 800:
                '''Few-shot classification for target domain data set'''
                # get few-shot classification samples
                task = utils.Task(target_da_metatrain_data, TEST_CLASS_NUM, SHOT_NUM_PER_CLASS, QUERY_NUM_PER_CLASS)
                support_dataloader = utils.get_HBKC_data_loader(task, num_per_class=SHOT_NUM_PER_CLASS, split="train", shuffle=False)
                query_dataloader = utils.get_HBKC_data_loader(task, num_per_class=QUERY_NUM_PER_CLASS, split="test", shuffle=True)
                
                supports, support_labels = support_dataloader.__iter__().__next__()
                querys, query_labels = query_dataloader.__iter__().__next__()
                
                supports = to_cuda(supports)
                querys = to_cuda(querys)
                support_proto_features, query_Enhance_features, per_outputs, enhance_loss = feature_encoder(supports, querys, support_labels.cuda(), query_labels.cuda(), domain='target')
                # fsl_loss
                logits = MD_distance_0(support_proto_features, per_outputs, support_labels, query_Enhance_features)
                f_loss = crossEntropy(logits, query_labels.cuda())
                loss = f_loss + enhance_loss

                # Update parameters
                feature_encoder_optim.zero_grad()
                loss.backward()
                feature_encoder_optim.step()

                total_hit += torch.sum(torch.argmax(logits, dim=1).cpu() == query_labels).item()
                total_num += query_labels.shape[0]

            if (episode + 1) % 100 == 0:
                train_loss.append(loss.item())
                print('episode {:>3d}: loss: {:6.4f}, query_sample_num: {:>3d}, acc {:6.4f}'.format(i * EPISODE_1 + episode + 1, \
                                                                                                                loss.item(),
                                                                                                                query_labels.shape[0],
                                                                                                                total_hit / total_num))
            if (episode + 1) % 1000 == 0 or flag == 1:
                # test
                print("Testing ...")
                feature_encoder.eval()
                total_rewards = 0
                counter = 0
                accuracies = []
                predict = np.array([], dtype=np.int64)
                labels = np.array([], dtype=np.int64)
                test_features_all = []
                test_labels_all = np.array([], dtype=np.int64)

                train_datas, train_labels = train_loader.__iter__().__next__()

                for test_datas, test_labels in test_loader:
                    train_datas = to_cuda(train_datas)
                    test_datas = to_cuda(test_datas)
                    train_proto_features, test_enhance_features, per_outputs, _= feature_encoder(train_datas, test_datas, train_labels.cuda(), None, domain='target')

                    batch_size = test_labels.shape[0]  # Dataloader set batch_size 128
                    predict_logits = MD_distance_0(train_proto_features, per_outputs, train_labels, test_enhance_features)
            
                    test_features_tmp = test_enhance_features.cpu().detach().numpy()
                    test_features_all.append(test_features_tmp)

                    predict_labels = torch.argmax(predict_logits, dim=1).cpu()
                    test_labels = test_labels.numpy()
                    total_rewards += np.sum([1 if predict_labels[j] == test_labels[j] else 0 for j in range(batch_size)])

                    counter += batch_size

                    predict = np.append(predict, predict_labels)
                    labels = np.append(labels, test_labels)

                    accuracy = total_rewards / 1.0 / counter  #
                    accuracies.append(accuracy)
                test_accuracy = 100. * total_rewards / len(test_loader.dataset)

                print('\t\tAccuracy: {}/{} ({:.2f}%) iDataSet: {} \n'.format( total_rewards, len(test_loader.dataset),
                    100. * total_rewards / len(test_loader.dataset), iDataSet))    
                # Training mode
                feature_encoder.train()
                if test_accuracy > last_accuracy:
                    # save networks
                    torch.save(feature_encoder.state_dict(),str(CHECKPOINT_PATH + "/feature_encoder_" +str(iDataSet) +"iter_" + str(TEST_LSAMPLE_NUM_PER_CLASS) +"shot.pkl"))
                    print("save networks for episode:",episode+1)
                    last_accuracy = test_accuracy
                    best_episdoe =  i * EPISODE_1 + episode 

                    acc[iDataSet] = total_rewards / len(test_loader.dataset)
                    OA = acc[iDataSet]
                    C = metrics.confusion_matrix(labels, predict)
                    A[iDataSet, :] = np.diag(C) / np.sum(C, 1, dtype=np.float64)
                    P[iDataSet, :] = np.diag(C) / np.sum(C, 1, dtype=np.float64)

                    k[iDataSet] = metrics.cohen_kappa_score(labels, predict)

                print('best episode:[{}], best accuracy={}'.format(best_episdoe + 1, last_accuracy))
                flag = 0

    training_time[iDataSet] = 0
    test_time[iDataSet] = 0

    latest_G, latest_RandPerm, latest_Row, latest_Column, latest_nTrain = G, RandPerm, Row, Column, nTrain
    for i in range(len(predict)):
        latest_G[latest_Row[latest_RandPerm[latest_nTrain + i]]][latest_Column[latest_RandPerm[latest_nTrain + i]]] = \
            predict[i] + 1
    sio.savemat(CLASSIFICATIONMAP_PATH + '/pred_map_latest' + '_' + str(iDataSet) + "iter_" + repr(int(OA * 10000)) + '.mat', {'latest_G': latest_G})

    if test_accuracy > best_acc_all:
        best_predict_all = predict
        best_G,best_RandPerm,best_Row, best_Column,best_nTrain = G, RandPerm, Row, Column, nTrain
    print('iter:{} best episode:[{}], best accuracy={}'.format(iDataSet, best_episdoe + 1, last_accuracy))
    print('***********************************************************************************')
###
ELEMENT_ACC_RES_SS4 = np.transpose(A)
AA_RES_SS4 = np.mean(ELEMENT_ACC_RES_SS4,0)
OA_RES_SS4 = np.transpose(acc)
KAPPA_RES_SS4 = np.transpose(k)
ELEMENT_PRE_RES_SS4 = np.transpose(P)
AP_RES_SS4= np.mean(ELEMENT_PRE_RES_SS4,0)
TRAINING_TIME_RES_SS4 = 0
TESTING_TIME_RES_SS4 = np.transpose(test_time)
classes_num = TEST_CLASS_NUM

outputs_chart = OutputData(classes_num, N_RUNS)
for current_trial_turn in range(N_RUNS):
    outputs_chart.set_data('train_time', current_trial_turn, training_time[current_trial_turn])
    outputs_chart.set_data('predict_time', current_trial_turn, test_time[current_trial_turn])
    outputs_chart.set_data('AA', current_trial_turn, np.around(AA_RES_SS4[current_trial_turn] * 100, 2))
    outputs_chart.set_data('OA', current_trial_turn, np.around(acc[current_trial_turn] * 100, 2))
    outputs_chart.set_data('Kappa', current_trial_turn, np.around(k[current_trial_turn] * 100, 2))
    for i in range(1, classes_num + 1):
        outputs_chart.set_data(i, current_trial_turn, np.around(A[current_trial_turn][i - 1] * 100, 2))

if not os.path.isdir(RESULT_DIR):
    os.makedirs(RESULT_DIR)
SAVE_PATH = RESULT_DIR + str(N_RUNS) + "_Twopatch517_MD5_" + str(EPISODE) + "_" + str(TEST_LSAMPLE_NUM_PER_CLASS) + "shot" + ".xlsx"
xlsxname = 'New_work'
outputs_chart.output_data(SAVE_PATH, xlsxname)