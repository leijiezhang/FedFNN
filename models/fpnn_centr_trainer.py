import copy
import torch
import wandb
from torch import nn
from models.losses import BalancedSoftmax


class CentralizedTrainer(object):
    r"""
    This class is used to train federated non-IID dataset in a centralized way
    """

    def __init__(self, dataset, model, rules_idx_list, args):
        self.device = args.device
        self.args = args
        train_data_global, test_data_global, _, _ = \
            dataset.get_local_data()
        self.train_global = train_data_global
        self.test_global = test_data_global
        self.rules_idx_list = rules_idx_list
        self.model = model
        self.device = args.device

    def train(self):
        metrics_list = []
        for epoch in range(self.args.epochs):
            if self.args.data_parallel == 1:
                self.train_global.sampler.set_epoch(epoch)
            self.train_impl(epoch, self.device)
            metrics_rtn = self.eval_impl(epoch)
            metrics_list.append(metrics_rtn)
        return metrics_list

    def train_impl(self, epoch_idx, p_device):
        self.model.to(p_device)
        if self.args.criterion == "ce":
            criterion = torch.nn.CrossEntropyLoss().to(p_device)
        elif self.args.criterion == "bce":
            criterion = BalancedSoftmax().to(p_device)
        else:
            criterion = BalancedSoftmax().to(p_device)

        if self.args.optimizer == "sgd":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.lr)
        elif self.args.optimizer == "adam":
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.lr,
                                         weight_decay=self.args.wd, amsgrad=True)
        else:
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.lr)
        self.model.train()
        batch_loss = []
        for batch_idx, (x, labels) in enumerate(self.train_global):
            # logging.info(images.shape)
            x, labels = x.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            log_probs, _ = self.model(x)
            loss = criterion(log_probs, labels.squeeze())
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())
        epoch_loss = sum(batch_loss) / len(batch_loss)
        self.args.logger.info('Epoch: {}\tLoss: {:.6f}'.format(
            epoch_idx, epoch_loss))

    def eval_impl(self, epoch_idx):
        train_metrics = self._global_test(b_train=True)
        train_num = copy.deepcopy(train_metrics['test_total'])
        train_correct_num = copy.deepcopy(train_metrics['test_correct'])
        train_loss = copy.deepcopy(train_metrics['test_loss'])
        train_acc = train_correct_num / train_num
        train_loss = train_loss / train_num

        train_rule_fs = copy.deepcopy(train_metrics['fs'])

        # update local training information
        _, train_fs_max = torch.max(train_rule_fs, -1)
        # count the numbers of samples that treat certain rule as their best rule
        train_rule_count = torch.nn.functional.one_hot(train_fs_max).sum(0)
        if train_rule_count.shape[0] < self.args.n_rule:
            n_diff = self.args.n_rule - train_rule_count.shape[0]
            arr_diff = torch.zeros(n_diff).long().to(self.args.device)
            train_rule_count = torch.cat([train_rule_count, arr_diff], 0)
        train_rule_contr = train_rule_fs.mean(0)

        # test data
        test_metrics = self._global_test(b_train=True)
        test_num = copy.deepcopy(test_metrics['test_total'])
        test_correct_num = copy.deepcopy(test_metrics['test_correct'])
        test_loss = copy.deepcopy(test_metrics['test_loss'])
        test_acc = test_correct_num / test_num
        test_loss = test_loss / test_num

        test_rule_fs = copy.deepcopy(test_metrics['fs'])

        # update local testing information
        _, test_fs_max = torch.max(test_rule_fs, -1)
        # count the numbers of samples that treat certain rule as their best rule
        test_rule_count = torch.nn.functional.one_hot(test_fs_max).sum(0)
        if test_rule_count.shape[0] < self.args.n_rule:
            n_diff = self.args.n_rule - test_rule_count.shape[0]
            arr_diff = torch.zeros(n_diff).long().to(self.args.device)
            test_rule_count = torch.cat([test_rule_count, arr_diff], 0)
        test_rule_contr = test_rule_fs.mean(0)

        metrics_stats = {'training_acc': train_acc, 'training_loss': train_loss,
                         'test_acc': test_acc, 'test_loss': test_loss}

        metrics_rule = {}
        for rule_idx in torch.arange(self.args.n_rule):
            metrics_rule[f"rule{rule_idx + 1}_count"] = int(train_rule_count[rule_idx])
            metrics_rule[f"rule{rule_idx + 1}_contr"] = float(train_rule_contr[rule_idx])

        metrics_rtn = {**metrics_stats, **metrics_rule}
        if not self.args.b_debug:
            wandb.log(metrics_rtn)
        self.args.logger.info(metrics_stats)
        return metrics_rtn

    def _test_on_global_client(self, test_data, p_device):
        model = self.model

        model.to(p_device)
        model.eval()

        metrics_normal = {
            'test_correct': 0,
            'test_loss': 0,
            'test_total': 0
        }

        fs = torch.empty(0, self.args.n_rule).to(p_device)
        # criterion = torch.nn.CrossEntropyLoss().to(device)
        if self.args.criterion == "ce":
            criterion = torch.nn.CrossEntropyLoss().to(p_device)
        elif self.args.criterion == "bce":
            criterion = BalancedSoftmax().to(p_device)
        else:
            criterion = BalancedSoftmax().to(p_device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(p_device)
                target = target.to(p_device)
                pred, fire_strength = model(x)
                loss = criterion(pred, target.squeeze())

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target.squeeze()).sum()

                fs_tmp = torch.zeros(fire_strength.shape[0], self.args.n_rule).to(self.args.device)
                fs_tmp[:, self.rules_idx_list] = fire_strength
                fs = torch.cat([fs, fs_tmp], 0)

                metrics_normal['test_correct'] += correct.item()
                metrics_normal['test_loss'] += loss.item() * target.size(0)
                metrics_normal['test_total'] += target.size(0)

        metrics_normal['fs'] = fs
        return metrics_normal

    def _global_test(self, b_train):
        self.model.update_rules_idx_list(self.rules_idx_list)
        if b_train:
            test_data = self.train_global
        else:
            test_data = self.test_global
        metrics = self._test_on_global_client(test_data, self.device)
        return metrics
