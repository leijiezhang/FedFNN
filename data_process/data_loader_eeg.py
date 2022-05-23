import numpy as np
import scipy.io as sio
import logging
import torch.utils.data as data
from .dataset_eeg import Data_truncated


logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    print('Data statistics: %s' % str(net_cls_counts))
    return net_cls_counts


def get_dataloader(X_train, y_train, X_test, y_test, datadir, train_bs, test_bs, dataidxs=None):
    dl_obj = Data_truncated

    # transform_train, transform_test = None

    train_ds = dl_obj(datadir, X_train, y_train, X_test, y_test,
                      dataidxs=dataidxs, train=True, transform=None, download=True)
    test_ds = dl_obj(datadir, X_train, y_train, X_test, y_test,
                     train=False, transform=None, download=True)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)

    return train_dl, test_dl


def load_partition_data(n_client, data_dir, batch_size):
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = load_data(n_client, data_dir)

    class_num = len(np.unique(y_train))
    logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = sum([len(net_dataidx_map[r]) for r in range(n_client)])

    train_data_global, test_data_global = get_dataloader(X_train, y_train, X_test, y_test,
                                                         data_dir, batch_size, batch_size)
    logging.info("train_dl_global number = " + str(len(train_data_global)))
    logging.info("test_dl_global number = " + str(len(test_data_global)))
    test_data_num = len(test_data_global)

    # get local dataset
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()

    for client_idx in range(n_client):
        dataidxs = net_dataidx_map[client_idx]
        local_data_num = len(dataidxs)
        data_local_num_dict[client_idx] = local_data_num
        logging.info("client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num))

        # training batch size = 64; algorithms batch size = 32
        train_data_local, test_data_local = get_dataloader(X_train, y_train, X_test, y_test,
                                                           data_dir, batch_size, batch_size,
                                                           dataidxs)
        logging.info("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
            client_idx, len(train_data_local), len(test_data_local)))
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local
    return train_data_num, test_data_num, train_data_global, test_data_global, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num


def load_data(n_subject, data_path='../data/xiaofei/'):
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    net_dataidx_map = {}

    for i in range(n_subject):
        test_data = sio.loadmat(data_path + '/subj' + str(i + 1) + '.mat')['test_data']
        X_test.append(test_data)
        test_label = sio.loadmat(data_path + '/subj' + str(i + 1) + '.mat')['test_label'].squeeze()
        y_test.append(test_label)
        train_data = sio.loadmat(data_path + '/subj' + str(i + 1) + '.mat')['train_data']
        X_train.append(train_data)
        train_label = sio.loadmat(data_path + '/subj' + str(i + 1) + '.mat')['train_label'].squeeze()
        y_train.append(train_label)
        train_idx = sio.loadmat(data_path + '/subj' + str(i + 1) + '.mat')['train_idx'].squeeze()
        net_dataidx_map[i] = train_idx

    X_train = tuple(X_train)
    y_train = tuple(y_train)
    X_train = np.concatenate(X_train, axis=0)
    X_train = X_train.astype(np.float32)
    y_train = np.concatenate(y_train, axis=0)
    y_train = y_train.astype(np.int64)

    X_test = tuple(X_test)
    y_test = tuple(y_test)
    X_test = np.concatenate(X_test, axis=0)
    X_test = X_test.astype(np.float32)
    y_test = np.concatenate(y_test, axis=0)
    y_test = y_test.astype(np.int64)

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)

    return X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts


if __name__ == "__main__":
    subject_list = [1, 2, 3, 5, 6, 7, 8, 9]
    load_data(subject_list)