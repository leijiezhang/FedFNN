from data_process.partition import *
from utils.math_utils import mapminmax
from torch.utils.data import Dataset as Dataset_nn
import scipy.io as sio
import numpy as np
import torch.utils.data as data


class Dataset(data.Dataset):
    def __init__(self, data, target, data_idxs=None,
                 transform=None, target_transform=None):
        self.data_idxs = data_idxs
        self.transform = transform
        self.target_transform = target_transform

        self.data, self.target = self.__build_truncated_dataset__(data, target)

    def __build_truncated_dataset__(self, data_para, target_para):
        data_return = data_para
        target_return = target_para
        if self.data_idxs is not None:
            data_return = data_para[self.data_idxs]
            target_return = target_para[self.data_idxs]
        return data_return, target_return

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        data_return, target = self.data[index], self.target[index]

        if self.transform is not None:
            data_return = self.transform(data_return)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return data_return, target

    def __len__(self):
        return len(self.data)


class FedDatasetCV(data.Dataset):
    """
        This dataset is designed for cross validation
        we suppose the data structure is X: N x D (N is the number of data samples and D is the data sample dimention)
        and the label set Y as: N x 1
    """

    def __init__(self, inputs, targets, n_class, task, name, p_args=None, data_idxs=None,
                 transform=None, target_transform=None):
        """
        init the Dataset class
        :param inputs: the features of data
        :param targets: the ground true label for classification or regression task
        :param name: the name of data set
        :param task: R for regression C for classification
        """
        self.name = name
        self.task = task

        self.n_class = n_class

        # data sequance disorder
        self.shuffle = True

        self.args = p_args

        self.data_idxs = data_idxs
        self.transform = transform
        self.target_transform = target_transform

        self.inputs, self.targets = self.__build_truncated_dataset__(inputs, targets)
        self.n_fea = self.inputs.shape[1]
        self.n_smpl = self.inputs.shape[0]

        # partition dataset into several test data and training data
        self.current_fold = 0
        self.fed_kfold_partition = []
        if self.args is not None:
            # init partition strategy
            fed_kfold_partition = FedKfoldPartition(self.args)
            fed_kfold_partition.partition(targets, self.shuffle, 0)
            self.fed_kfold_partition = fed_kfold_partition

    def __build_truncated_dataset__(self, data_para, target_para):
        data_return = data_para
        target_return = target_para
        if self.data_idxs is not None:
            data_return = data_para[self.data_idxs]
            target_return = target_para[self.data_idxs].squeeze()
        return data_return, target_return

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        data_return, target = self.inputs[index], self.targets[index]

        if self.transform is not None:
            data_return = self.transform(data_return)

        if self.target_transform is not None:
            target = self.target_transform(target).squeeze()

        return data_return, target

    def __len__(self):
        return len(self.inputs)

    def set_current_folds(self, cur_fold):
        self.current_fold = cur_fold

    def set_partition(self, fed_kfold_partition: FedKfoldPartition):
        self.fed_kfold_partition = fed_kfold_partition

    def set_shuffle(self, shuffle):
        self.shuffle = shuffle

    def get_local_data(self, fold_idx=None):
        """
        todo:generate training dataset and test dataset by k-folds
        :param fold_idx:
        :return: 1st fold datasets for run by default or specified n fold runabel datasets
        """
        assert self.args is not None
        if fold_idx is not None:
            self.set_current_folds(fold_idx)
        # get data loader dict for local clients
        train_data_local_dict = dict()
        test_data_local_dict = dict()
        for test_local_i, test_local_dataidx_i in enumerate(self.fed_kfold_partition.local_test_indexes):
            local_test_idx = test_local_dataidx_i[self.current_fold]
            local_test_name = f"client{test_local_i}_{self.name}_train_cv{self.current_fold}"
            local_targets_test = FedDatasetCV(inputs=self.inputs, targets=self.targets, n_class=self.n_class,
                                              task=self.task, name=local_test_name, data_idxs=local_test_idx,
                                              transform=None)
            batch_size = self.args.batch_size
            if batch_size > local_test_idx.shape[0]:
                batch_size = local_test_idx.shape[0]
            local_targets_test_dl = data.DataLoader(dataset=local_targets_test, batch_size=batch_size,
                                                    shuffle=True, drop_last=True)
            test_data_local_dict[test_local_i] = local_targets_test_dl

        for train_local_i, train_local_dataidx_i in enumerate(self.fed_kfold_partition.local_train_indexes):
            local_train_idx = train_local_dataidx_i[self.current_fold]
            local_train_name = f"client{train_local_i}_{self.name}_train_cv{self.current_fold}"
            local_targets_train = FedDatasetCV(inputs=self.inputs, targets=self.targets, n_class=self.n_class,
                                               task=self.task, name=local_train_name, data_idxs=local_train_idx,
                                               transform=None)
            batch_size = self.args.batch_size
            if batch_size > local_train_idx.shape[0]:
                batch_size = local_train_idx.shape[0]
            local_targets_train_dl = data.DataLoader(dataset=local_targets_train, batch_size=batch_size,
                                                     shuffle=True, drop_last=True)
            train_data_local_dict[train_local_i] = local_targets_train_dl

        # get data loader dict for centralized method
        global_train_idx = self.fed_kfold_partition.global_train_indexes[self.current_fold]
        global_train_name = f"{self.name}_train_cv{self.current_fold}"
        train_data_global = FedDatasetCV(inputs=self.inputs, targets=self.targets, n_class=self.n_class,
                                         task=self.task, name=global_train_name, data_idxs=global_train_idx,
                                         transform=None)
        train_data_global_dl = data.DataLoader(dataset=train_data_global, batch_size=self.args.batch_size,
                                               shuffle=True, drop_last=True)

        global_test_idx = self.fed_kfold_partition.global_test_indexes[self.current_fold]
        global_test_name = f"{self.name}_test_cv{self.current_fold}"
        test_data_global = FedDatasetCV(inputs=self.inputs, targets=self.targets, n_class=self.n_class,
                                        task=self.task, name=global_test_name, data_idxs=global_test_idx,
                                        transform=None)
        test_data_global_dl = data.DataLoader(dataset=test_data_global, batch_size=self.args.batch_size,
                                              shuffle=True, drop_last=True)

        return train_data_global_dl, test_data_global_dl, train_data_local_dict, test_data_local_dict

    def get_class_cound_list(self):
        local_test_cls_counts_list = []
        local_train_cls_counts_list = []
        if self.args is not None:
            local_test_cls_counts_list = self.fed_kfold_partition.local_test_cls_counts_list[self.current_fold]
            local_train_cls_counts_list = self.fed_kfold_partition.local_train_cls_counts_list[self.current_fold]
        return local_test_cls_counts_list, local_train_cls_counts_list

    def get_local_saml_num(self):
        local_test_saml_num_list = []
        local_train_saml_num_list = []
        if self.args is not None:
            local_test_cls_counts_list = self.fed_kfold_partition.local_test_cls_counts_list[self.current_fold]
            local_train_cls_counts_list = self.fed_kfold_partition.local_train_cls_counts_list[self.current_fold]
            for client_i in range(len(local_train_cls_counts_list)):
                local_test_saml_num_list.append(sum(local_test_cls_counts_list[client_i].values()))
                local_train_saml_num_list.append(sum(local_train_cls_counts_list[client_i].values()))
        return local_test_saml_num_list, local_train_saml_num_list


def get_dataset_mat(dir_dataset, p_args):
    # dir_dataset = f"./data/{dataset_name}/{dataset_name}.mat"

    load_data = sio.loadmat(dir_dataset)

    # inputs: torch.Tensor = load_data['X']).float().to(device)
    # targets: torch.Tensor = torch.tensor(load_data['Y'].astype(np.float32)).float().to(device)
    inputs = load_data['inputs'].astype(np.float32)
    # normalize data
    if p_args.b_norm_dataset:
        inputs = mapminmax(inputs)
    if p_args.nl > 0.0:
        element_num = inputs.shape[0] * inputs.shape[1]
        # # element wise
        noise_num = int(p_args.nl * element_num)
        mu, sigma = 0, 0.8  # mean and standard deviation
        noise = np.random.normal(mu, sigma, element_num).reshape((inputs.shape[0], inputs.shape[1]))

        mask = np.zeros((element_num, 1))
        mask[0:noise_num, :] = 1
        mask = mask[np.random.permutation(element_num), :].reshape((inputs.shape[0], inputs.shape[1]))
        mask = mask == 1
        inputs[mask] = noise[mask] + inputs[mask]
    targets = load_data['targets'].astype(np.int64)
    targets = targets - targets.min()
    n_class = int(targets.max() + 1)

    if len(targets.shape) == 1:
        targets = targets.unsqueeze(1)

    task = str(load_data['task'])

    # init partition strategy
    partition_strategy = FedKfoldPartition(p_args)
    partition_strategy.partition(targets, True, 0)

    dataset = FedDatasetCV(inputs, targets, n_class, task, p_args.dataset, p_args=p_args)
    # set partition strategy
    dataset.set_partition(partition_strategy)

    # dataset.normalize(-1, 1)
    return dataset
