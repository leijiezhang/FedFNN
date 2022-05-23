import copy
import random
import numpy as np
import torch
import wandb
from data_process.dataset import FedDatasetCV
from models.client import Client


class FedAvgAPI(object):
    def __init__(self, dataset: FedDatasetCV, model_trainer, p_args):
        self.args = p_args
        train_data_global, test_data_global, train_data_local_dict, test_data_local_dict = \
            dataset.get_local_data()
        self.test_global = test_data_global
        self.val_global = None

        self.client_list = []
        _, self.train_data_local_class_count = dataset.get_class_cound_list()
        _, self.train_data_local_num_dict = dataset.get_local_saml_num()
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict

        self.model_trainer = model_trainer
        self._setup_clients(self.train_data_local_num_dict, self.train_data_local_class_count, train_data_local_dict,
                            test_data_local_dict, model_trainer)

    def _setup_clients(self, train_data_local_num_dict, train_data_local_class_count,
                       train_data_local_dict, test_data_local_dict, model_trainer):
        self.args.logger.info("############setup_clients (START)#############")
        for client_idx in range(self.args.n_client_per_round):
            c = Client(client_idx, train_data_local_dict[client_idx], test_data_local_dict[client_idx],
                       train_data_local_num_dict[client_idx], train_data_local_class_count[client_idx],
                       self.args, self.args.device, model_trainer)
            self.client_list.append(c)
        self.args.logger.info("############setup_clients (END)#############")

    def train(self):
        w_global = self.model_trainer.get_model_params()
        metrics_list = []
        for round_idx in range(self.args.comm_round):

            self.args.logger.info("################Communication round : {}".format(round_idx))

            w_locals = []

            """
            for scalability: following the original FedAvg algorithm, we uniformly sample a fraction of clients in each round.
            Instead of changing the 'Client' instances, our implementation keeps the 'Client' instances and then updates their local dataset 
            """
            client_indexes = self._client_sampling(round_idx, self.args.n_client,
                                                   self.args.n_client_per_round)
            self.args.logger.info("client_indexes = " + str(client_indexes))

            for idx, client in enumerate(self.client_list):
                # update dataset
                client_idx = client_indexes[idx]
                client.update_local_dataset(client_idx, self.train_data_local_dict[client_idx],
                                            self.test_data_local_dict[client_idx],
                                            self.train_data_local_num_dict[client_idx],
                                            self.train_data_local_class_count[client_idx])

                # train on new dataset
                w = client.train(copy.deepcopy(w_global))
                # self.logger.info("local weights = " + str(w))
                w_locals.append((client.get_sample_number(), copy.deepcopy(w)))

            # update global weights
            w_global = self._aggregate(w_locals)
            self.model_trainer.set_model_params(w_global)

            # test results
            metrics_rtn = self._local_test_on_all_clients(round_idx)
            # # at last round
            # if round_idx == self.args.comm_round - 1:
            #     self._local_test_on_all_clients(round_idx)
            # # per {frequency_of_the_test} round
            # elif round_idx % self.args.frequency_of_the_test == 0:
            #     if self.args.dataset.startswith("stackoverflow"):
            #         self._local_test_on_validation_set(round_idx)
            #     else:
            #         self._local_test_on_all_clients(round_idx)
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
        training_num = 0
        for idx in range(len(w_locals)):
            (sample_num, averaged_params) = w_locals[idx]
            training_num += sample_num

        (sample_num, averaged_params) = w_locals[0]
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
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

        client = self.client_list[0]

        for client_idx in range(self.args.n_client):
            """
            Note: for datasets like "fed_CIFAR100" and "fed_shakespheare",
            the training client number is larger than the testing client number
            """
            if self.test_data_local_dict[client_idx] is None:
                continue
            client.update_local_dataset(0, self.train_data_local_dict[client_idx],
                                        self.test_data_local_dict[client_idx],
                                        self.train_data_local_num_dict[client_idx],
                                            self.train_data_local_class_count[client_idx])
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
        return metrics_rtn

    def _local_test_on_validation_set(self, round_idx):

        self.args.logger.info("################local_test_on_validation_set : {}".format(round_idx))

        if self.val_global is None:
            self._generate_validation_set()

        client = self.client_list[0]
        client.update_local_dataset(0, None, self.val_global, None)
        # test data
        test_metrics = client.local_test(True)

        if self.args.dataset == "stackoverflow_nwp":
            test_acc = test_metrics['test_correct'] / test_metrics['test_total']
            test_loss = test_metrics['test_loss'] / test_metrics['test_total']
            stats = {'test_acc': test_acc, 'test_loss': test_loss}
            wandb.log({"Test/Acc": test_acc, "round": round_idx})
            wandb.log({"Test/Loss": test_loss, "round": round_idx})
        elif self.args.dataset == "stackoverflow_lr":
            test_acc = test_metrics['test_correct'] / test_metrics['test_total']
            test_pre = test_metrics['test_precision'] / test_metrics['test_total']
            test_rec = test_metrics['test_recall'] / test_metrics['test_total']
            test_loss = test_metrics['test_loss'] / test_metrics['test_total']
            stats = {'test_acc': test_acc, 'test_pre': test_pre, 'test_rec': test_rec, 'test_loss': test_loss}
            wandb.log({"Test/Acc": test_acc, "round": round_idx})
            wandb.log({"Test/Pre": test_pre, "round": round_idx})
            wandb.log({"Test/Rec": test_rec, "round": round_idx})
            wandb.log({"Test/Loss": test_loss, "round": round_idx})
        else:
            raise Exception("Unknown format to log metrics for dataset {}!" % self.args.dataset)

        self.args.logger.info(stats)
