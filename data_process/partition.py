import abc
import numpy as np


class KFoldPartition(object):
    def __init__(self, n_fold):
        super(KFoldPartition, self).__init__()
        self.current_fold = 0
        self.train_indexes = []
        self.test_indexes = []
        self.num_folds = n_fold

    def partition(self, gnd, is_shuffle=True, random_state=None):
        n_smpl = gnd.shape[0]
        total_index = np.arange(n_smpl)
        if is_shuffle:
            if random_state is not None:
                np.random.seed(random_state)

            total_index = np.random.permutation(n_smpl)

        n_test_smpl = round(n_smpl / self.num_folds)
        self.train_indexes = []
        self.test_indexes = []
        for i in np.arange(self.num_folds):
            test_start = n_test_smpl * i
            test_end = n_test_smpl * (i + 1)

            test_index = total_index[test_start:(test_end - 1)]

            if test_start != 0:
                train_index_part1 = total_index[0:test_start - 1]
                train_index_part2 = total_index[test_end::]
                train_index = np.concatenate((train_index_part1, train_index_part2), axis=0)
            else:
                train_index = total_index[test_end::]

            self.train_indexes.append(train_index)
            self.test_indexes.append(test_index)
        self.set_current_folds(0)

    def get_train_indexes(self):
        train_idx = self.train_indexes[self.current_fold]
        num_train_idx = train_idx.shape[0]
        return train_idx, num_train_idx

    def get_test_indexes(self):
        test_idx = self.test_indexes[self.current_fold]
        num_test_idx = test_idx.shape[0]
        return test_idx, num_test_idx

    def get_description(self):
        d = ('%i-fold partition', self.num_folds)
        return d

    def get_num_folds(self):
        return self.num_folds

    def set_current_folds(self, cur_fold):
        self.current_fold = cur_fold


class FederatedPartition(object):
    def __init__(self, n_client, partition_type, partition_alpha):
        super(FederatedPartition, self).__init__()
        self.n_client = n_client
        self.partition_type = partition_type
        self.local_smpl_idxs = []
        self.test_indexes = []
        self.partition_alpha = partition_alpha

    def partition(self, gnd, is_shuffle=True, random_state=None):
        net_dataidx_map = {}
        if self.partition_type == "homo":
            total_num = gnd.shape[0]
            idxs = np.random.permutation(total_num)
            batch_idxs = np.array_split(idxs, self.n_client)
            for i in range(self.n_client):
                self.local_smpl_idxs.append(batch_idxs[i])

        elif self.partition_type == "hetero":
            min_size = 0
            class_num = np.unique(gnd).shape[0]
            N = gnd.shape[0]

            idx_batch = []
            while min_size < 10:
                idx_batch = [[] for _ in range(self.n_client)]
                # for each class in the dataset
                for class_i in range(class_num):
                    class_i_idx = np.where(gnd == class_i)[0]
                    np.random.shuffle(class_i_idx)
                    proportions = np.random.dirichlet(np.repeat(self.partition_alpha, self.n_client))
                    ## Balance
                    proportions = np.array([p * (len(idx_j) < N / self.n_client) for p, idx_j in zip(proportions, idx_batch)])
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(class_i_idx)).astype(int)[:-1]
                    idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(class_i_idx, proportions))]
                    min_size = min([len(idx_j) for idx_j in idx_batch])

            for j in range(self.n_client):
                np.random.shuffle(idx_batch[j])
                self.local_smpl_idxs.append(idx_batch[j])


class FedKfoldPartition(object):
    def __init__(self, p_args):
        super(FedKfoldPartition, self).__init__()
        self.args = p_args
        self.local_train_indexes = []
        self.local_test_indexes = []
        self.global_train_indexes = []
        self.global_test_indexes = []
        self.local_test_cls_counts_list = []
        self.local_train_cls_counts_list = []

    def partition(self, gnd, is_shuffle=True, random_state=None):
        n_smpl = gnd.shape[0]
        total_index_origin = np.arange(n_smpl)
        total_index_shuffled = np.arange(n_smpl)
        gnd_origin = gnd
        gnd_shuffled = gnd.copy()
        if is_shuffle:
            if random_state is not None:
                np.random.seed(random_state)
            shuf_index = np.random.permutation(n_smpl)
            total_index_shuffled = total_index_origin[shuf_index]
            gnd_shuffled = gnd_origin[total_index_shuffled].copy()
        idx_batch = []
        if self.args.partition_method == "homo":
            total_num = gnd.shape[0]
            idxs = np.random.permutation(total_num)
            batch_idxs = np.array_split(idxs, self.args.n_client)
            for i in range(self.args.n_client):
                idx_batch.append(batch_idxs[i].tolist())

        elif self.args.partition_method == "hetero":
            class_num = np.unique(gnd).shape[0]
            n_smpl = gnd.shape[0]

            idx_batch = [[] for _ in range(self.args.n_client)]
            # for each class in the dataset
            for class_i in range(class_num):
                class_i_idx = total_index_shuffled[np.where(gnd_shuffled == class_i)[0]]
                # np.random.shuffle(class_i_idx)
                proportions = np.random.dirichlet(np.repeat(self.args.partition_alpha, self.args.n_client))
                ## Balance
                proportions = np.array([p * (len(idx_j) < n_smpl / self.args.n_client)
                                        for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(class_i_idx)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(class_i_idx, proportions))]

        # kfold for local client
        for client_i in range(self.args.n_client):
            if is_shuffle:
                np.random.shuffle(idx_batch[client_i])
            local_idx = idx_batch[client_i]
            n_smpl_local = len(local_idx)
            n_test_loca = round(n_smpl_local / self.args.n_kfolds)
            local_train_indexes = []
            local_test_indexes = []
            for kfold_i in np.arange(self.args.n_kfolds):
                local_test_start = n_test_loca * kfold_i
                local_test_end = n_test_loca * (kfold_i + 1)

                local_test_index = np.array(local_idx[local_test_start:(local_test_end - 1)])

                if local_test_start != 0:
                    local_train_index_part1 = local_idx[0:local_test_start - 1]
                    local_train_index_part2 = local_idx[local_test_end::]
                    local_train_index = np.array(local_train_index_part1 + local_train_index_part2)
                else:
                    local_train_index = np.array(local_idx[local_test_end::])

                local_train_indexes.append(local_train_index)
                local_test_indexes.append(local_test_index)

            self.local_train_indexes.append(local_train_indexes)
            self.local_test_indexes.append(local_test_indexes)

        # kfold global for centralized method
        for kfold_j in np.arange(self.args.n_kfolds):
            global_train_kfold_j = np.empty([0], dtype=int)
            global_test_kfold_j = np.empty([0], dtype=int)
            for client_j in range(self.args.n_client):
                global_train_kfold_j = np.concatenate((global_train_kfold_j,
                                                           self.local_train_indexes[client_j][kfold_j]), axis=0)
                global_test_kfold_j = np.concatenate((global_test_kfold_j,
                                                          self.local_test_indexes[client_j][kfold_j]), axis=0)
            self.global_train_indexes.append(global_train_kfold_j)
            self.global_test_indexes.append(global_test_kfold_j)

        self.record_local_data_stats(gnd_origin)

    def record_local_data_stats(self, gnd):
        for cv_i in np.arange(self.args.n_kfolds):
            local_test_cls_counts_cv_i = {}
            for test_local_i, test_local_dataidx_i in enumerate(self.local_test_indexes):
                unq, unq_cnt = np.unique(gnd[test_local_dataidx_i[cv_i]], return_counts=True)
                tmp = {unq[i]: unq_cnt[i] for i in range(unq.shape[0])}
                local_test_cls_counts_cv_i[test_local_i] = tmp
            self.local_test_cls_counts_list.append(local_test_cls_counts_cv_i)

        for cv_i in np.arange(self.args.n_kfolds):
            local_train_cls_counts_cv_i = {}
            for train_local_i, train_local_dataidx_i in enumerate(self.local_train_indexes):
                unq, unq_cnt = np.unique(gnd[train_local_dataidx_i[cv_i]], return_counts=True)
                tmp = {unq[i]: unq_cnt[i] for i in range(unq.shape[0])}
                local_train_cls_counts_cv_i[train_local_i] = tmp
            self.local_train_cls_counts_list.append(local_train_cls_counts_cv_i)
