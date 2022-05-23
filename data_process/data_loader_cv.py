import logging
import scipy.io as sio
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import random
from .datasets_cv import Data
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


# generate the non-IID distribution for all methods
def read_data_distribution(filename='./data_preprocessing/non-iid-distribution/cv/distribution.txt'):
    distribution = {}
    with open(filename, 'r') as data:
        for x in data.readlines():
            if '{' != x[0] and '}' != x[0]:
                tmp = x.split(':')
                if '{' == tmp[1].strip():
                    first_level_key = int(tmp[0])
                    distribution[first_level_key] = {}
                else:
                    second_level_key = int(tmp[0])
                    distribution[first_level_key][second_level_key] = int(tmp[1].strip().replace(',', ''))
    return distribution


def read_net_dataidx_map(filename='./data_preprocessing/non-iid-distribution/cv/net_dataidx_map.txt'):
    net_dataidx_map = {}
    with open(filename, 'r') as data:
        for x in data.readlines():
            if '{' != x[0] and '}' != x[0] and ']' != x[0]:
                tmp = x.split(':')
                if '[' == tmp[-1].strip():
                    key = int(tmp[0])
                    net_dataidx_map[key] = []
                else:
                    tmp_array = x.split(',')
                    net_dataidx_map[key] = [int(i.strip()) for i in tmp_array]
    return net_dataidx_map


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    logging.debug('Data statistics: %s' % str(net_cls_counts))
    return net_cls_counts


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cv():
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    train_transform.transforms.append(Cutout(16))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    return train_transform, valid_transform


def load_cv_data(datadir, cv_idx=1, noise_level=0.0):
    inputs = sio.loadmat(datadir + '/inputs.mat')['inputs']
    labels = np.squeeze(sio.loadmat(datadir + '/labels.mat')['targets'])
    cv_idx_cur = sio.loadmat(datadir + '/cv_idx' + str(cv_idx) + '.mat')

    cv_idx_train = np.squeeze(cv_idx_cur['train_idx'])
    cv_idx_test = np.squeeze(cv_idx_cur['test_idx'])

    # transform_train, transform_test = None
    train_ds = Data(inputs, labels,
                    data_idxs=cv_idx_train-1, transform=None)
    test_ds = Data(inputs, labels,
                   data_idxs=cv_idx_test-1, transform=None)

    X_train, y_train = train_ds.data.astype(np.float32), train_ds.target.astype(np.int64)
    X_test, y_test = test_ds.data.astype(np.float32), test_ds.target.astype(np.int64)

    if noise_level > 0.0:
        n_smpl = X_train.shape[0]
        n_fea = X_train.shape[1]
        element_num = n_smpl * n_fea

        noise = np.random.normal(0, 0.8, element_num).reshape((n_smpl, n_fea))
        
        # element wise
        noise_num = int(noise_level * element_num)
        mask = np.zeros((element_num, 1))
        mask[0:noise_num, :] = 1
        mask = mask[np.random.permutation(element_num), :].reshape((n_smpl, n_fea))
        mask = mask == 1

        # train_data.fea[mask] = 0.1*noise[mask] + train_data.fea[mask]
        X_train[mask] = noise[mask] + X_train[mask]

    return (X_train, y_train, X_test, y_test)


def partition_data(dataset, datadir, partition, n_nets, alpha, cv_idx=1, noise_level=0.0):
    logging.info("*********partition data***************")
    # load data
    X_train, y_train, X_test, y_test = load_cv_data(datadir, cv_idx=cv_idx, noise_level=noise_level)
    n_train = X_train.shape[0]
    # n_test = X_test.shape[0]

    if partition == "homo":
        total_num = n_train
        idxs = np.random.permutation(total_num)
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

    elif partition == "hetero":
        min_size = 0
        K = np.unique(y_train).shape[0]
        N = y_train.shape[0]
        logging.info("N = " + str(N))
        net_dataidx_map = {}

        while min_size < 10:
            idx_batch = [[] for _ in range(n_nets)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
                ## Balance
                proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    elif partition == "hetero-fix":
        dataidx_map_file_path = './data_preprocessing/non-iid-distribution/cv/net_dataidx_map.txt'
        net_dataidx_map = read_net_dataidx_map(dataidx_map_file_path)

    if partition == "hetero-fix":
        distribution_file_path = './data_preprocessing/non-iid-distribution/cv/distribution.txt'
        traindata_cls_counts = read_data_distribution(distribution_file_path)
    else:
        traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)

    return X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts


# for centralized training
def get_dataloader(dataset, inputs_train, labels_train, inputs_test, labels_test,
                   datadir, train_bs, test_bs, dataidxs=None):
    return get_dataloader_cv(datadir, inputs_train, labels_train,
           inputs_test, labels_test, train_bs, test_bs, dataidxs)


# for local devices
def get_dataloader_test(dataset, inputs_train, labels_train,
                        inputs_test, labels_test, datadir,
                        train_bs, test_bs, dataidxs_train,
                        dataidxs_test):
    return get_dataloader_test_cv(datadir, inputs_train, labels_train,
                                  inputs_test, labels_test, datadir,
                                  train_bs, test_bs, dataidxs_train, dataidxs_test)


def get_dataloader_cv(datadir, inputs_train, labels_train, inputs_test,
                      labels_test, train_bs, test_bs, dataidxs=None):
    dl_obj = Data

    # transform_train, transform_test = _data_transforms_cv()

    train_ds = dl_obj(inputs_train, labels_train, data_idxs=dataidxs, transform=None)
    test_ds = dl_obj(inputs_test, labels_test, data_idxs=None, transform=None)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)

    return train_dl, test_dl


def get_dataloader_test_cv(datadir, inputs_train, labels_train,
                           inputs_test, labels_test,
                           train_bs, test_bs, dataidxs,
                           train=None, dataidxs_test=None):
    dl_obj = Data

    # transform_train, transform_test = _data_transforms_cv()

    train_ds = dl_obj(inputs_train, labels_train, data_idxs=dataidxs, transform=None)
    test_ds = dl_obj(inputs_test, labels_test, data_idxs=dataidxs, transform=None)


    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)

    return train_dl, test_dl


def load_partition_data_distributed_cv(process_id, dataset, data_dir, partition_method, partition_alpha,
                                            client_number, batch_size, cv_idx=1):
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(dataset,
                                                                                             data_dir,
                                                                                             partition_method,
                                                                                             client_number,
                                                                                             partition_alpha,
                                                                                             cv_idx=cv_idx)
    class_num = len(np.unique(y_train))
    logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])

    # get global test data
    if process_id == 0:
        train_data_global, test_data_global = get_dataloader(dataset, data_dir, batch_size, batch_size)
        logging.info("train_dl_global number = " + str(len(train_data_global)))
        logging.info("test_dl_global number = " + str(len(test_data_global)))
        train_data_local = None
        test_data_local = None
        local_data_num = 0
    else:
        # get local dataset
        dataidxs = net_dataidx_map[process_id - 1]
        local_data_num = len(dataidxs)
        logging.info("rank = %d, local_sample_number = %d" % (process_id, local_data_num))
        # training batch size = 64; algorithms batch size = 32
        train_data_local, test_data_local = get_dataloader(dataset, data_dir, batch_size, batch_size,
                                                 dataidxs)
        logging.info("process_id = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
            process_id, len(train_data_local), len(test_data_local)))
        train_data_global = None
        test_data_global = None
    return train_data_num, train_data_global, test_data_global, local_data_num, train_data_local, test_data_local, class_num


def load_partition_data_cv(dataset, data_dir, partition_method, partition_alpha, client_number, batch_size,
                           cv_idx=1, noise_level=0.0):
    alpha = 0
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(dataset,
                                                                                             data_dir,
                                                                                             partition_method,
                                                                                             client_number,
                                                                                             partition_alpha,
                                                                                             cv_idx=cv_idx,
                                                                                             noise_level=noise_level)
    class_num = len(np.unique(y_train))
    logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])

    train_data_global, test_data_global = get_dataloader(dataset, inputs_train=X_train, labels_train=y_train,
                                                         inputs_test=X_test, labels_test=y_test,
                                                         datadir=data_dir, train_bs=batch_size, test_bs=batch_size,
                                                         dataidxs=None)
    logging.info("train_dl_global number = " + str(len(train_data_global)))
    logging.info("test_dl_global number = " + str(len(test_data_global)))
    test_data_num = len(test_data_global)

    # get local dataset
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()

    for client_idx in range(client_number):
        dataidxs = net_dataidx_map[client_idx]
        if alpha == 0:
            local_data_num = len(dataidxs)
            data_local_num_dict[client_idx] = local_data_num
            logging.info("client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num))

            # training batch size = 64; algorithms batch size = 32
            train_data_local, test_data_local = get_dataloader(dataset, inputs_train=X_train, labels_train=y_train,
                                                               inputs_test=X_test, labels_test=y_test,
                                                               datadir=data_dir, train_bs=batch_size,
                                                               test_bs=batch_size,
                                                               dataidxs=dataidxs)
            logging.info("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
                client_idx, len(train_data_local), len(test_data_local)))
        else:
            X_train_tmp = X_train[dataidxs]
            y_train_tmp = y_train[dataidxs]
            unq, unq_cnt = np.unique(y_train_tmp, return_counts=True)
            indices = np.argsort(unq_cnt)
            expected_number = unq_cnt[indices[int(len(unq)/2)+1]]
            X_train_tmp_gen = []
            y_train_tmp_gen = []
            for k in range(len(unq)):
                idx_k = np.where(y_train_tmp == indices[k])[0]

                X_train_tmp_tmp = X_train_tmp[idx_k]
                y_train_tmp_tmp = y_train_tmp[idx_k]
                if k==0:
                    X_train_tmp_gen = X_train_tmp_tmp
                    y_train_tmp_gen = y_train_tmp_tmp
                else:
                    X_train_tmp_gen = np.concatenate((X_train_tmp_gen, X_train_tmp_tmp))
                    y_train_tmp_gen = np.concatenate((y_train_tmp_gen, y_train_tmp_tmp))
                if k <= int(len(unq)/2):
                    local_num = len(idx_k)
                    div_num = expected_number - local_num
                    expand_times = int(np.rint(div_num / local_num))
                    for kk in range(expand_times):
                        lam = np.random.beta(alpha, alpha)
                        index_tmp = np.random.permutation(local_num)
                        mixed_x = lam * X_train_tmp_tmp + (1 - lam) * X_train_tmp[index_tmp, :]
                        X_train_tmp_gen = np.concatenate((X_train_tmp_gen, mixed_x))
                        y_train_tmp_gen = np.concatenate((y_train_tmp_gen, y_train_tmp_tmp))
            unq_gen, unq_cnt_gen = np.unique(y_train_tmp_gen, return_counts=True)
            idx_gen = np.arange(len(y_train_tmp_gen))
            np.random.shuffle(idx_gen)
            X_train_gen = X_train_tmp_gen[idx_gen]
            y_train_gen = y_train_tmp_gen[idx_gen]
            train_data_local, test_data_local = get_dataloader(dataset, inputs_train=X_train_gen, labels_train=y_train_gen,
                                                               inputs_test=X_test, labels_test=y_test,
                                                               datadir=data_dir, train_bs=batch_size,
                                                               test_bs=batch_size,
                                                               dataidxs=None)
            logging.info("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
                client_idx, len(train_data_local), len(test_data_local)))


        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local
    return train_data_num, test_data_num, train_data_global, test_data_global, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num
