import copy
import random
import numpy as np
import torch
import wandb
from data_process.dataset import FedDatasetCV
from models.client import Client, FedFPNNClient
from models.fpnn import *


class FedAvgAPI(object):
    def __init__(self, dataset: FedDatasetCV, global_model, p_args):
        self.args = p_args
        train_data_global, test_data_global, train_data_local_dict, test_data_local_dict = \
            dataset.get_local_data()
        self.test_global = test_data_global
        self.val_global = None

        self.client_list = []
        self.client_rule_list = []
        _, self.train_data_local_class_count = dataset.get_class_cound_list()
        _, self.train_data_local_num_dict = dataset.get_local_saml_num()
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict

        self.global_model = global_model

        self._setup_clients(self.train_data_local_num_dict, self.train_data_local_class_count, train_data_local_dict,
                            test_data_local_dict)
        self.rules_client_dict = {}

    def _setup_clients(self, train_data_local_num_dict, train_data_local_class_count,
                       train_data_local_dict, test_data_local_dict):
        self.args.logger.info("############setup_clients (START)#############")
        for client_idx in range(self.args.n_client):
            # till now the data won't need any update. It is fixed
            local_rules_idx_list = copy.deepcopy(self.global_model.rules_idx_list)
            c = FedFPNNClient(client_idx, local_rules_idx_list,
                              train_data_local_dict[client_idx], test_data_local_dict[client_idx],
                              train_data_local_num_dict[client_idx], train_data_local_class_count[client_idx],
                              self.args, self.global_model)
            self.client_rule_list.append(self.global_model.rules_idx_list)
            self.client_list.append(c)
        self.args.logger.info("############setup_clients (END)#############")

    def get_rules_client_dict(self):
        for rule_i in range(self.args.n_rule):
            self.rules_client_dict[rule_i] = []
            for client_j in range(self.args.n_client):
                if rule_i in self.client_list[client_j].local_rule_idxs:
                    self.rules_client_dict[rule_i].append(client_j)

    def train(self):
        w_global = self.global_model.cpu().state_dict()
        metrics_list = []
        for round_idx in range(self.args.comm_round):

            self.args.logger.info("################Communication round : {}".format(round_idx))

            w_locals = []

            """
            for scalability: following the original FedAvg algorithm, we uniformly sample a fraction of clients in each round.
            Instead of changing the 'Client' instances, our implementation keeps the 'Client' instances and then updates their local dataset 
            """
            client_indexes_per_round = self._client_sampling(round_idx, self.args.n_client,
                                                             self.args.n_client_per_round)
            self.args.logger.info("client_indexes = " + str(client_indexes_per_round))

            for client_idx in client_indexes_per_round:
                client = self.client_list[client_idx]
                # train on local client
                w = client.train(w_global)
                # self.args.logger.info("local weights = " + str(w))
                w_locals.append((client.local_sample_number, copy.deepcopy(w)))

            # update global weights
            w_global = self._aggregate(w_locals)
            self.global_model.update_model(w_global)
            # test results
            # # at last round
            # if round_idx == self.args.comm_round - 1:
            #     self._local_test_on_all_clients(round_idx)
            # # per {frequency_of_the_test} round
            # elif round_idx % self.args.frequency_of_the_test == 0:
            #     self._local_test_on_all_clients(round_idx)

            metrics_rtn = self._local_test_on_all_clients(round_idx)
            metrics_list.append(metrics_rtn)
        return metrics_list

    def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        self.args.logger.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def _generate_validation_set(self, num_samples=10000):
        test_data_num = len(self.test_global.dataset)
        sample_indices = random.sample(range(test_data_num), min(num_samples, test_data_num))
        subset = torch.utils.data.Subset(self.test_global.dataset, sample_indices)
        sample_testset = torch.utils.data.DataLoader(subset, batch_size=self.args.batch_size)
        self.val_global = sample_testset

    def _aggregate(self, w_locals):
        averaged_params = self.global_model.cpu().state_dict()
        # aggregate the firing strength layer
        # training_num_fs = 0
        # for i in range(0, len(w_locals)):
        #     local_sample_number, _ = w_locals[i]
        #     training_num_fs += local_sample_number
        #
        # for k in averaged_params.keys():
        #     for i in range(0, len(w_locals)):
        #         local_sample_number_fs, local_model_params = w_locals[i]
        #         if f"rule" not in k:
        #             w = local_sample_number_fs / training_num_fs
        #             if i == 0:
        #                 averaged_params[k] = local_model_params[k] * w
        #             else:
        #                 averaged_params[k] += local_model_params[k] * w

        # aggregate the rule parameters
        for k in averaged_params.keys():
            if "rule" in k:
                rule_training_num = 0
                for idx in range(len(w_locals)):
                    (sample_num, _) = w_locals[idx]
                    if int(k.split(".")[0].split("_")[1]) in self.client_list[idx].rules_idx_list:
                        rule_training_num += sample_num
                tag_j = 0
                for idx in range(len(w_locals)):
                    local_sample_number, local_model_params = w_locals[idx]
                    w = local_sample_number / rule_training_num
                    if int(k.split(".")[0].split("_")[1]) in self.client_list[idx].rules_idx_list:
                        if tag_j == 0:
                            averaged_params[k] = local_model_params[k] * w
                            tag_j = 1
                        else:
                            averaged_params[k] += local_model_params[k] * w
            else:
                training_num_fs = 0
                for i in range(0, len(w_locals)):
                    local_sample_number, _ = w_locals[i]
                    training_num_fs += local_sample_number
                for i in range(0, len(w_locals)):
                    local_sample_number_fs, local_model_params = w_locals[i]
                    w = local_sample_number_fs / training_num_fs
                    if i == 0:
                        averaged_params[k] = local_model_params[k] * w
                    else:
                        averaged_params[k] += local_model_params[k] * w

        # for rule_i in self.global_model.local_rule_idxs:
        #     rule_training_num = 0
        #     for idx in range(len(w_locals)):
        #         (sample_num, _) = w_locals[idx]
        #         if f"local_rule_{rule_i}.antecedent_layer.proto" in averaged_params.keys():
        #             rule_training_num += sample_num
        #
        #     tag_j = 0
        #     for client_j in range(0, len(w_locals)):
        #         local_sample_number, local_model_params = w_locals[client_j]
        #         if f"local_rule_{rule_i}.antecedent_layer.proto" in averaged_params.keys():
        #             w = local_sample_number / rule_training_num
        #             if tag_j == 0:
        #                 averaged_params[f"local_rule_{rule_i}.antecedent_layer.proto"] = \
        #                     local_model_params[f"local_rule_{rule_i}.antecedent_layer.proto"] * w
        #                 averaged_params[f"local_rule_{rule_i}.antecedent_layer.var"] = \
        #                     local_model_params[f"local_rule_{rule_i}.antecedent_layer.var"] * w
        #                 averaged_params[f"local_rule_{rule_i}.consequent_layer.consq_layers.weight"] = \
        #                     local_model_params[f"local_rule_{rule_i}.consequent_layer.consq_layers.weight"] * w
        #                 averaged_params[f"local_rule_{rule_i}.consequent_layer.consq_layers.bias"] = \
        #                     local_model_params[f"local_rule_{rule_i}.consequent_layer.consq_layers.bias"] * w
        #                 tag_j += 1
        #             else:
        #                 averaged_params[f"local_rule_{rule_i}.antecedent_layer.proto"] += \
        #                     local_model_params[f"local_rule_{rule_i}.antecedent_layer.proto"] * w
        #                 averaged_params[f"local_rule_{rule_i}.antecedent_layer.var"] += \
        #                     local_model_params[f"local_rule_{rule_i}.antecedent_layer.var"] * w
        #                 averaged_params[f"local_rule_{rule_i}.consequent_layer.consq_layers.weight"] += \
        #                     local_model_params[f"local_rule_{rule_i}.consequent_layer.consq_layers.weight"] * w
        #                 averaged_params[f"local_rule_{rule_i}.consequent_layer.consq_layers.bias"] += \
        #                     local_model_params[f"local_rule_{rule_i}.consequent_layer.consq_layers.bias"] * w

        # self.global_model.update_model(averaged_params)
        return averaged_params

    def _local_test_on_all_clients(self, round_idx):

        self.args.logger.info("################local_test_on_all_clients : {}".format(round_idx))

        metrics = {
            'train_num_samples': [],
            'train_num_correct': [],
            'train_losses': [],
            'test_num_samples': [],
            'test_num_correct': [],
            'test_losses': [],
        }

        train_rule_count_local = torch.zeros(self.args.n_client, self.args.n_rule).to(self.args.device)
        test_rule_count_local = torch.zeros(self.args.n_client, self.args.n_rule).to(self.args.device)
        train_rule_contr_local = torch.zeros(self.args.n_client, self.args.n_rule).to(self.args.device)
        test_rule_contr_local = torch.zeros(self.args.n_client, self.args.n_rule).to(self.args.device)

        train_acc_local = torch.zeros(self.args.n_client).to(self.args.device)
        test_acc_local = torch.zeros(self.args.n_client).to(self.args.device)
        train_loss_local = torch.zeros(self.args.n_client).to(self.args.device)
        test_loss_local = torch.zeros(self.args.n_client).to(self.args.device)

        train_rule_fs = torch.empty(0, self.args.n_rule).to(self.args.device)
        test_rule_fs = torch.empty(0, self.args.n_rule).to(self.args.device)

        # client = self.client_list[0]

        for client_idx in range(self.args.n_client):
            """
            Note: for datasets like "fed_CIFAR100" and "fed_shakespheare",
            the training client number is larger than the testing client number
            """
            if self.test_data_local_dict[client_idx] is None:
                continue
            # client.update_local_dataset(0, self.train_data_local_dict[client_idx],
            #                             self.test_data_local_dict[client_idx],
            #                             self.train_data_local_num_dict[client_idx],
            #                             self.train_data_local_class_count[client_idx])
            # train data
            client = self.client_list[client_idx]

            #train data
            train_local_metrics = client.local_test(False)
            train_num_client = copy.deepcopy(train_local_metrics['test_total'])
            train_correct_num_client = copy.deepcopy(train_local_metrics['test_correct'])
            train_loss_all_client = copy.deepcopy(train_local_metrics['test_loss'])
            metrics['train_num_samples'].append(train_num_client)
            metrics['train_num_correct'].append(train_correct_num_client)
            metrics['train_losses'].append(train_loss_all_client)
            train_acc_client = train_correct_num_client / train_num_client
            train_loss_client = train_loss_all_client / train_num_client
            train_acc_local[client_idx] = train_acc_client
            train_loss_local[client_idx] = train_loss_client

            train_rule_fs_client = copy.deepcopy(train_local_metrics['fs'])
            train_rule_fs = torch.cat((train_rule_fs, train_rule_fs_client), 0)

            # update local training information
            _, train_fs_max_client = torch.max(train_rule_fs_client, -1)
            # count the numbers of samples that treat certain rule as their best rule
            train_rule_count_client = torch.nn.functional.one_hot(train_fs_max_client).sum(0)
            if train_rule_count_client.shape[0] < self.args.n_rule:
                n_diff = self.args.n_rule - train_rule_count_client.shape[0]
                arr_diff = torch.zeros(n_diff).long().to(self.args.device)
                train_rule_count_client = torch.cat([train_rule_count_client, arr_diff], 0)
            train_rule_contr_client = train_rule_fs_client.mean(0)
            self.args.logger.info(f"training rule contribution on client{client_idx}: {train_rule_contr_client}")
            train_rule_count_local[client_idx, :] = train_rule_count_client
            train_rule_contr_local[client_idx, :] = train_rule_contr_client

            # test data
            test_local_metrics = client.local_test(True)
            test_num_client = copy.deepcopy(test_local_metrics['test_total'])
            test_correct_num_client = copy.deepcopy(test_local_metrics['test_correct'])
            test_loss_all_client = copy.deepcopy(test_local_metrics['test_loss'])
            metrics['test_num_samples'].append(test_num_client)
            metrics['test_num_correct'].append(test_correct_num_client)
            metrics['test_losses'].append(test_loss_all_client)

            # test on local test dataset
            test_acc_client = test_correct_num_client / test_num_client
            test_loss_client = test_loss_all_client / test_num_client
            test_acc_local[client_idx] = test_acc_client
            test_loss_local[client_idx] = test_loss_client

            test_rule_fs_client = copy.deepcopy(test_local_metrics['fs'])
            test_rule_fs = torch.cat((test_rule_fs, test_rule_fs_client), 0)

            # update local training information
            _, test_fs_max_client = torch.max(test_rule_fs_client, -1)
            # count the numbers of samples that treat certain rule as their best rule
            test_rule_count_client = torch.nn.functional.one_hot(test_fs_max_client).sum(0)
            if test_rule_count_client.shape[0] < self.args.n_rule:
                n_diff = self.args.n_rule - test_rule_count_client.shape[0]
                arr_diff = torch.zeros(n_diff).long().to(self.args.device)
                test_rule_count_client = torch.cat([test_rule_count_client, arr_diff], 0)
            test_rule_contr_client = test_rule_fs_client.mean(0)
            test_rule_count_local[client_idx, :] = test_rule_count_client
            test_rule_contr_local[client_idx, :] = test_rule_contr_client

            # change the rule list when reaching milestones
            # if (round_idx + 1) % self.args.milestone == 0 and round_idx < self.args.milestone:
            if (round_idx + 1) % self.args.milestone == 0:
                train_local_metrics = client.local_test(False)
                train_num_client = copy.deepcopy(train_local_metrics['test_total'])
                train_correct_num_client = copy.deepcopy(train_local_metrics['test_correct'])
                train_loss_all_client = copy.deepcopy(train_local_metrics['test_loss'])
                train_acc_client = train_correct_num_client / train_num_client
                train_loss_client = train_loss_all_client / train_num_client

                train_rule_fs_client = copy.deepcopy(train_local_metrics['fs'])
                train_rule_fs = torch.cat((train_rule_fs, train_rule_fs_client), 0)

                # update local training information
                _, train_fs_max_client = torch.max(train_rule_fs_client, -1)
                # count the numbers of samples that treat certain rule as their best rule
                train_rule_count_client = torch.nn.functional.one_hot(train_fs_max_client).sum(0)
                if train_rule_count_client.shape[0] < self.args.n_rule:
                    n_diff = self.args.n_rule - train_rule_count_client.shape[0]
                    arr_diff = torch.zeros(n_diff).long().to(self.args.device)
                    train_rule_count_client = torch.cat([train_rule_count_client, arr_diff], 0)
                train_rule_contr_client = train_rule_fs_client.mean(0)
                c_th = np.arange(self.args.n_rule)[train_rule_contr_client.cpu() > 1 / client.rules_idx_list.size]
                # c_th = np.arange(self.args.n_rule)[train_rule_contr_client.cpu() > 0.7 / self.args.n_rule]
                # c_th = np.arange(self.args.n_rule)[train_rule_contr_client.cpu() > 1e-3]
                # c_th = c_th[train_rule_count_client[c_th].cpu() > 150]
                if client.rules_idx_list.size >= client.n_rule_limit:
                    client.update_rule_idx_list(c_th)

        if (round_idx + 1) % self.args.milestone == 0 and round_idx < 100:
            explore_rule_idx = torch.arange(self.args.n_client)[train_acc_local < train_acc_local.mean()]
            _, indices = torch.sort(train_acc_local[explore_rule_idx], descending=False)
            explore_rule_idx = explore_rule_idx[indices]
            activate_rule_tag = torch.zeros(self.args.n_rule, dtype=torch.bool)
            for client_idx in range(self.args.n_client):
                rules_idx_list_tmp = self.client_list[client_idx].rules_idx_list
                activate_rule_tag[rules_idx_list_tmp] = True
            for update_client_idx in explore_rule_idx:
                if (activate_rule_tag == False).sum() > 0 \
                        and self.client_list[update_client_idx].rules_idx_list.size <= \
                        self.client_list[update_client_idx].n_rule_limit\
                        and self.client_list[update_client_idx].rules_idx_list.size <= \
                        (self.args.n_rule_max - 1):
                    activate_rule_idx = torch.arange(self.args.n_rule)[activate_rule_tag == False][0]
                    self.client_list[update_client_idx].rules_idx_list = \
                        np.insert(self.client_list[update_client_idx].rules_idx_list, 
                                  self.client_list[update_client_idx].rules_idx_list.size, 
                                  activate_rule_idx)
                    self.client_list[update_client_idx].rules_idx_list.sort()
                    self.client_list[update_client_idx].n_rule_limit += 1
                    activate_rule_tag[activate_rule_idx] = True
        _, train_fs_max = torch.max(train_rule_fs, -1)
        # count the numbers of samples that treat certain rule as their best rule
        train_rule_count = torch.nn.functional.one_hot(train_fs_max).sum(0)
        if train_rule_count.shape[0] < self.args.n_rule:
            n_diff = self.args.n_rule - train_rule_count.shape[0]
            arr_diff = torch.zeros(n_diff).long().to(self.args.device)
            train_rule_count = torch.cat([train_rule_count, arr_diff], 0)
        # train_rule_count_sum = train_rule_count / train_rule_count.sum()

        _, test_fs_max = torch.max(test_rule_fs, -1)
        test_rule_count = torch.nn.functional.one_hot(test_fs_max).sum(0)
        if test_rule_count.shape[0] < self.args.n_rule:
            n_diff = self.args.n_rule - test_rule_count.shape[0]
            arr_diff = torch.zeros(n_diff).long().to(self.args.device)
            test_rule_count = torch.cat([test_rule_count, arr_diff], 0)
        # test_rule_count_sum = test_rule_count / test_rule_count.sum()

        # get the contribution of each rule, namely the averaged fs on all samples
        train_rule_contr = train_rule_fs.mean(0)
        test_rule_contr = test_rule_fs.mean(0)

        metrics_rule = {}
        for rule_idx in torch.arange(self.args.n_rule):
            metrics_rule[f"rule{rule_idx + 1}_count"] = int(train_rule_count[rule_idx])
            metrics_rule[f"rule{rule_idx + 1}_contr"] = float(train_rule_contr[rule_idx])
            # train rule information
            # metrics_rule[f"train_rule{rule_idx + 1}_count"] = float(train_rule_count[rule_idx])
            # metrics_rule[f"train_rule{rule_idx + 1}_contr"] = float(train_rule_contr[rule_idx])
            # test rule information
            # metrics_rule[f"test_rule{rule_idx + 1}_count"] = float(test_rule_count[rule_idx])
            # metrics_rule[f"test_rule{rule_idx + 1}_contr"] = float(test_rule_cndt_sum[rule_idx])
            for client_idx in range(self.args.n_client):
                metrics_rule[f"client{client_idx + 1}_rule{rule_idx + 1}_count"] = int(
                    train_rule_count_local[client_idx, rule_idx])
                metrics_rule[f"client{client_idx + 1}_rule{rule_idx + 1}_contr"] = float(
                    train_rule_contr_local[client_idx, rule_idx])
                metrics_rule[f"client{client_idx + 1}_train_acc"] = float(
                    train_acc_local[client_idx])
                metrics_rule[f"client{client_idx + 1}_train_loss"] = float(
                    train_loss_local[client_idx])
                metrics_rule[f"client{client_idx + 1}_test_acc"] = float(
                    test_acc_local[client_idx])
                metrics_rule[f"client{client_idx + 1}_test_loss"] = float(
                    test_loss_local[client_idx])

        # test on training dataset
        train_acc = sum(metrics['train_num_correct']) / sum(metrics['train_num_samples'])
        train_loss = sum(metrics['train_losses']) / sum(metrics['train_num_samples'])

        # test on test dataset
        test_acc = sum(metrics['test_num_correct']) / sum(metrics['test_num_samples'])
        test_loss = sum(metrics['test_losses']) / sum(metrics['test_num_samples'])

        metrics_stats = {'training_acc': train_acc, 'training_loss': train_loss,
                         'test_acc': test_acc, 'test_loss': test_loss}

        metrics_rtn = {**metrics_stats, **metrics_rule}
        if not self.args.b_debug:
            wandb.log(metrics_rtn)
        self.args.logger.info(metrics_stats)
        for client_idx in range(self.args.n_client):
            self.args.logger.info(f"client {client_idx} ==> training_acc: {train_acc_local[client_idx]}, "
                                  f"training_loss: {train_loss_local[client_idx]}, "
                                  f"test_acc: {test_acc_local[client_idx]}, "
                                  f"test_loss: {test_loss_local[client_idx]}")
        return metrics_rtn

    def _eval_rules_on_all_clients(self, round_idx):

        self.args.logger.info("###############evaluate_contributions_of_rules_on_all_clients : {}".format(round_idx))

        metrics = {
            'train_num_samples': [],
            'train_num_correct': [],
            'train_losses': [],
            'test_num_samples': [],
            'test_num_correct': [],
            'test_losses': [],
        }

        train_rule_count_local = torch.zeros(self.args.n_client, self.args.n_rule).to(self.args.device)
        test_rule_count_local = torch.zeros(self.args.n_client, self.args.n_rule).to(self.args.device)
        train_rule_contr_local = torch.zeros(self.args.n_client, self.args.n_rule).to(self.args.device)
        test_rule_contr_local = torch.zeros(self.args.n_client, self.args.n_rule).to(self.args.device)

        train_acc_local = torch.zeros(self.args.n_client).to(self.args.device)
        test_acc_local = torch.zeros(self.args.n_client).to(self.args.device)
        train_loss_local = torch.zeros(self.args.n_client).to(self.args.device)
        test_loss_local = torch.zeros(self.args.n_client).to(self.args.device)

        train_rule_fs = torch.empty(0, self.args.n_rule).to(self.args.device)
        test_rule_fs = torch.empty(0, self.args.n_rule).to(self.args.device)

        client = self.client_list[0]

        for client_idx in range(self.args.n_client):
            """
            Note: for datasets like "fed_CIFAR100" and "fed_shakespheare",
            the training client number is larger than the testing client number
            """
            if self.test_data_local_dict[client_idx] is None:
                continue
            # train data
            train_local_metrics = client.local_test(False)
            train_num_client = copy.deepcopy(train_local_metrics['test_total'])
            train_correct_num_client = copy.deepcopy(train_local_metrics['test_correct'])
            train_loss_all_client = copy.deepcopy(train_local_metrics['test_loss'])
            metrics['train_num_samples'].append(train_num_client)
            metrics['train_num_correct'].append(train_correct_num_client)
            metrics['train_losses'].append(train_loss_all_client)
            train_acc_client = train_correct_num_client / train_num_client
            train_loss_client = train_loss_all_client / train_num_client
            train_acc_local[client_idx] = train_acc_client
            train_loss_local[client_idx] = train_loss_client

            train_rule_fs_client = copy.deepcopy(train_local_metrics['fs'])
            train_rule_fs = torch.cat((train_rule_fs, train_rule_fs_client), 0)

            # update local training information
            _, train_fs_max_client = torch.max(train_rule_fs_client, -1)
            # count the numbers of samples that treat certain rule as their best rule
            train_rule_count_client = torch.nn.functional.one_hot(train_fs_max_client).sum(0)
            if train_rule_count_client.shape[0] < self.args.n_rule:
                n_diff = self.args.n_rule - train_rule_count_client.shape[0]
                arr_diff = torch.zeros(n_diff).long().to(self.args.device)
                train_rule_count_client = torch.cat([train_rule_count_client, arr_diff], 0)
            train_rule_contr_client = train_rule_fs_client.mean(0)
            train_rule_count_local[client_idx, :] = train_rule_count_client
            train_rule_contr_local[client_idx, :] = train_rule_contr_client

            # test data
            test_local_metrics = client.local_test(True)
            test_num_client = copy.deepcopy(test_local_metrics['test_total'])
            test_correct_num_client = copy.deepcopy(test_local_metrics['test_correct'])
            test_loss_all_client = copy.deepcopy(test_local_metrics['test_loss'])
            metrics['test_num_samples'].append(test_num_client)
            metrics['test_num_correct'].append(test_correct_num_client)
            metrics['test_losses'].append(test_loss_all_client)

            # test on local test dataset
            test_acc_client = test_correct_num_client / test_num_client
            test_loss_client = test_loss_all_client / test_num_client
            test_acc_local[client_idx] = test_acc_client
            test_loss_local[client_idx] = test_loss_client

            test_rule_fs_client = copy.deepcopy(test_local_metrics['fs'])
            test_rule_fs = torch.cat((test_rule_fs, test_rule_fs_client), 0)

            # update local training information
            _, test_fs_max_client = torch.max(test_rule_fs_client, -1)
            # count the numbers of samples that treat certain rule as their best rule
            test_rule_count_client = torch.nn.functional.one_hot(test_fs_max_client).sum(0)
            if test_rule_count_client.shape[0] < self.args.n_rule:
                n_diff = self.args.n_rule - test_rule_count_client.shape[0]
                arr_diff = torch.zeros(n_diff).long().to(self.args.device)
                test_rule_count_client = torch.cat([test_rule_count_client, arr_diff], 0)
            test_rule_contr_client = test_rule_fs_client.mean(0)
            test_rule_count_local[client_idx, :] = test_rule_count_client
            test_rule_contr_local[client_idx, :] = test_rule_contr_client

        _, train_fs_max = torch.max(train_rule_fs, -1)
        # count the numbers of samples that treat certain rule as their best rule
        train_rule_count = torch.nn.functional.one_hot(train_fs_max).sum(0)
        if train_rule_count.shape[0] < self.args.n_rule:
            n_diff = self.args.n_rule - train_rule_count.shape[0]
            arr_diff = torch.zeros(n_diff).long().to(self.args.device)
            train_rule_count = torch.cat([train_rule_count, arr_diff], 0)
        # train_rule_count_sum = train_rule_count / train_rule_count.sum()

        _, test_fs_max = torch.max(test_rule_fs, -1)
        test_rule_count = torch.nn.functional.one_hot(test_fs_max).sum(0)
        if test_rule_count.shape[0] < self.args.n_rule:
            n_diff = self.args.n_rule - test_rule_count.shape[0]
            arr_diff = torch.zeros(n_diff).long().to(self.args.device)
            test_rule_count = torch.cat([test_rule_count, arr_diff], 0)
        # test_rule_count_sum = test_rule_count / test_rule_count.sum()

        # get the contribution of each rule, namely the averaged fs on all samples
        train_rule_contr = train_rule_fs.mean(0)
        test_rule_contr = test_rule_fs.mean(0)

        metrics_rule = {}
        for rule_idx in torch.arange(self.args.n_rule):
            metrics_rule[f"rule{rule_idx + 1}_count"] = int(train_rule_count[rule_idx])
            metrics_rule[f"rule{rule_idx + 1}_contr"] = float(train_rule_contr[rule_idx])
            # train rule information
            # metrics_rule[f"train_rule{rule_idx + 1}_count"] = float(train_rule_count[rule_idx])
            # metrics_rule[f"train_rule{rule_idx + 1}_contr"] = float(train_rule_contr[rule_idx])
            # test rule information
            # metrics_rule[f"test_rule{rule_idx + 1}_count"] = float(test_rule_count[rule_idx])
            # metrics_rule[f"test_rule{rule_idx + 1}_contr"] = float(test_rule_cndt_sum[rule_idx])
            for client_idx in range(self.args.n_client):
                metrics_rule[f"client{client_idx + 1}_rule{rule_idx + 1}_count"] = int(
                    train_rule_count_local[client_idx, rule_idx])
                metrics_rule[f"client{client_idx + 1}_rule{rule_idx + 1}_contr"] = float(
                    train_rule_contr_local[client_idx, rule_idx])
                metrics_rule[f"client{client_idx + 1}_train_acc"] = float(
                    train_acc_local[client_idx])
                metrics_rule[f"client{client_idx + 1}_train_loss"] = float(
                    train_loss_local[client_idx])
                metrics_rule[f"client{client_idx + 1}_test_acc"] = float(
                    test_acc_local[client_idx])
                metrics_rule[f"client{client_idx + 1}_test_loss"] = float(
                    test_loss_local[client_idx])

        # test on training dataset
        train_acc = sum(metrics['train_num_correct']) / sum(metrics['train_num_samples'])
        train_loss = sum(metrics['train_losses']) / sum(metrics['train_num_samples'])

        # test on test dataset
        test_acc = sum(metrics['test_num_correct']) / sum(metrics['test_num_samples'])
        test_loss = sum(metrics['test_losses']) / sum(metrics['test_num_samples'])

        metrics_stats = {'training_acc': train_acc, 'training_loss': train_loss,
                         'test_acc': test_acc, 'test_loss': test_loss}

        metrics_rtn = {**metrics_stats, **metrics_rule}
        if not self.args.b_debug:
            wandb.log(metrics_rtn)
        self.args.logger.info(metrics_stats)
        for client_idx in range(self.args.n_client):
            self.args.logger.info(f"client {client_idx} ==> training_acc: {train_acc_local[client_idx]}, "
                                  f"training_loss: {train_loss_local[client_idx]}, "
                                  f"test_acc: {test_acc_local[client_idx]}, "
                                  f"test_loss: {test_loss_local[client_idx]}")
