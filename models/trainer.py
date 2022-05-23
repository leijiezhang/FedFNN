import copy
import torch
import wandb
from torch import nn
from models.fpnn import FPNNnew
import torch.nn.functional as F


class NormalTrainer(object):
    r"""
    This class is used to train federated non-IID dataset in a centralized way
    """

    def __init__(self, train_loader, test_loader, model, args):
        self.device = args.device
        self.args = args
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.model = model
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        if self.args.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr)
        else:
            self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                              lr=self.args.lr,
                                              weight_decay=self.args.wd, amsgrad=True)

    def train(self):
        train_acc = torch.zeros(self.args.epochs, 1).to(self.args.device)
        test_acc = torch.zeros(self.args.epochs, 1).to(self.args.device)
        train_loss = torch.zeros(self.args.epochs, 1).to(self.args.device)
        test_loss = torch.zeros(self.args.epochs, 1).to(self.args.device)
        for epoch in range(self.args.epochs):
            self.train_impl(epoch)
            train_acc[epoch], train_loss[epoch], test_acc[epoch], test_loss[epoch] = self.eval_impl(epoch)
        return train_acc, train_loss, test_acc, test_loss

    def train_impl(self, epoch_idx):
        self.model.train()
        for batch_idx, (x, labels) in enumerate(self.train_loader):
            # self.args.logger.info(images.shape)
            x, labels = x.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            log_probs, _ = self.model(x)
            loss = self.criterion(log_probs, labels.squeeze())
            loss.backward()
            self.optimizer.step()
            self.args.logger.info('Local Training Epoch: {} {}-th iters\t Loss: {:.6f}'.format(epoch_idx,
                                                                                               batch_idx, loss.item()))

    def eval_impl(self, epoch_idx):
        train_acc = 0.0
        train_loss = 0.0
        test_acc = 0.0
        test_loss = 0.0

        # train
        # if epoch_idx % self.args.frequency_of_train_acc_report == 0:
        train_acc, train_loss, train_rule_contr = self.test(b_is_train=True, epoch_idx=epoch_idx)

        # test
        # if epoch_idx % self.args.frequency_of_test_acc_report == 0:
        test_acc, test_loss, test_rule_contr = self.test(b_is_train=False, epoch_idx=epoch_idx)

        log_train_rule_contr = {f"train_rule{rule_idx + 1}": float(train_rule_contr[rule_idx]) for rule_idx in 
                                torch.arange(train_rule_contr.shape[0])}
        log_test_rule_contr = {f"test_rule{rule_idx + 1}": float(test_rule_contr[rule_idx]) for rule_idx in
                                torch.arange(test_rule_contr.shape[0])}
        log_performance = {"epoch": epoch_idx, "Train Acc": train_acc, "Test Acc": test_acc,
                           "Train Loss": train_loss, "Test Loss": test_loss}
        log_report = {**log_train_rule_contr, **log_test_rule_contr, **log_performance}
        wandb.log(log_report)
        
        self.args.logger.debug(f"[epoch: {epoch_idx}, Train Acc: {train_acc}, Test Acc: {test_acc}, "
                               f"Train Loss: {train_loss}, Test Loss: {test_loss}")
        self.args.logger.debug(f"Train FS: {train_rule_contr.cpu()}")
        self.args.logger.debug(f"Test FS: {test_rule_contr.cpu()}")
        return train_acc, train_loss, test_acc, test_loss

    def test(self, b_is_train, epoch_idx):
        self.model.eval()
        contr_fs = torch.empty(0).int().to(self.args.device)
        metrics = {
            'test_correct': 0,
            'test_loss': 0,
            'test_precision': 0,
            'test_recall': 0,
            'test_total': 0
        }
        if b_is_train:
            test_data = self.train_loader
        else:
            test_data = self.test_loader
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(self.device)
                target = target.squeeze().to(self.device)
                pred, fire_strength = self.model(x)
                loss = self.criterion(pred, target.squeeze())

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                _, contr_fs_tmp = torch.max(fire_strength, -1)
                contr_fs = torch.cat([contr_fs, contr_fs_tmp], 0)

                metrics['test_correct'] += correct.item()
                metrics['test_loss'] += loss.item() * target.size(0)
                metrics['test_total'] += target.size(0)

        acc, loss = self.save_log(b_is_train=b_is_train, metrics=metrics)
        rule_contr = nn.functional.one_hot(contr_fs).sum(0)
        if rule_contr.shape[0] < self.args.n_rule:
            n_diff = self.args.n_rule - rule_contr.shape[0]
            arr_diff = torch.zeros(n_diff).long().to(self.args.device)
            rule_contr = torch.cat([rule_contr, arr_diff], 0)
        rule_contr = rule_contr / rule_contr.sum()
        return acc, loss, rule_contr

    def save_log(self, b_is_train, metrics):
        prefix = 'Train' if b_is_train else 'Test'

        all_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }

        all_metrics['num_samples'].append(copy.deepcopy(metrics['test_total']))
        all_metrics['num_correct'].append(copy.deepcopy(metrics['test_correct']))
        all_metrics['losses'].append(copy.deepcopy(metrics['test_loss']))

        # performance on all clients
        acc = sum(all_metrics['num_correct']) / sum(all_metrics['num_samples'])
        loss = sum(all_metrics['losses']) / sum(all_metrics['num_samples'])

        stats = {prefix + "/Loss": loss, prefix + '_acc': acc, prefix + '_loss': loss}
        # wandb.log({prefix + "/Acc": acc, "epoch": epoch_idx})
        # wandb.log({prefix + "/Loss": loss, "epoch": epoch_idx})

        # self.args.logger.debug(stats)
        return acc, loss
    
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)


class DynamicTrainer(object):
    def __init__(self, train_loader, test_loader, model, args):
        self.device = args.device
        self.args = args
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.model = model

        self.criterion = nn.CrossEntropyLoss()
        if self.args.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr)
        else:
            self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                              lr=self.args.lr,
                                              weight_decay=self.args.wd, amsgrad=True)

    def train(self):
        rule_contr = torch.ones(self.args.n_rule)
        # acc_cur = 0
        # acc_pre = 0
        train_acc = torch.zeros(self.args.epochs*self.args.update_round, 1).to(self.args.device)
        test_acc = torch.zeros(self.args.epochs*self.args.update_round, 1).to(self.args.device)
        train_loss = torch.zeros(self.args.epochs*self.args.update_round, 1).to(self.args.device)
        test_loss = torch.zeros(self.args.epochs*self.args.update_round, 1).to(self.args.device)

        for round_idx in range(self.args.update_round):

            self.args.logger.info("################ Update round : {}".format(round_idx))
            w_current = self.model.cpu().state_dict()
            # expand rule
            if round_idx > 0:
                # judge the useless rules
                useless_rule_idx = torch.where(rule_contr < 0.05)[0]
                normal_rule_idx = torch.where(rule_contr >= 0.05)[0]
                useful_rule_idx = torch.where(rule_contr >= 0.5)[0]

                n_useless_rule = useless_rule_idx.shape[0]
                n_normal_rule = normal_rule_idx.shape[0]
                n_useful_rule = useful_rule_idx.shape[0]
                n_expand_rule = n_useful_rule

                n_rule_new = n_normal_rule + n_expand_rule

                self.args.n_rule = n_rule_new
                prototype_list = torch.zeros(n_rule_new, self.args.n_fea)
                std = torch.zeros(n_rule_new, self.args.n_fea)

                model_new: torch.nn.Module = FPNNnew(prototype_list, std, self.args.n_class,
                                                      dropout_rate=self.args.dropout)
                w_new = model_new.state_dict()

                w_new["var"][0:n_normal_rule, :] = w_current["var"][normal_rule_idx, :]
                w_new["var"][n_normal_rule:n_rule_new, :] = w_current["var"][useful_rule_idx, :]
                w_new["proto"][0:n_normal_rule, :] = w_current["proto"][normal_rule_idx, :]
                for ii in torch.arange(n_expand_rule):
                    w_new["proto"][n_normal_rule:n_normal_rule + ii + 1, :] = torch.normal(
                        w_current["proto"][useful_rule_idx[ii], :],
                        self.model.var_process(w_current["var"][useful_rule_idx[ii], :]))
                for key, val in w_current.items():
                    if "fs_layers" in key:
                        w_new[key] = w_current[key]
                for jj in torch.arange(n_normal_rule):
                    new_idx = 'para_consq_{}'.format(jj)
                    current_idx = 'para_consq_{}'.format(normal_rule_idx[jj])
                    w_new[f"{new_idx}.weight"] = w_current[f"{current_idx}.weight"]
                    w_new[f"{new_idx}.bias"] = w_current[f"{current_idx}.bias"]
                for kk in torch.arange(n_useful_rule):
                    new_idx = 'para_consq_{}'.format(n_normal_rule + kk)
                    current_idx = 'para_consq_{}'.format(useful_rule_idx[kk])
                    w_new[f"{new_idx}.weight"] = w_current[f"{current_idx}.weight"]
                    w_new[f"{new_idx}.bias"] = w_current[f"{current_idx}.bias"]
                
                # model_new.load_state_dict(w_new)
                self.model = model_new
            self.model.to(self.args.device)
            if self.args.optimizer == "sgd":
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr)
            else:
                self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                                  lr=self.args.lr,
                                                  weight_decay=self.args.wd, amsgrad=True)
            for epoch in range(self.args.epochs):
                self.train_impl(epoch)
                train_acc[round_idx*self.args.update_round+epoch], \
                train_loss[round_idx*self.args.update_round+epoch], \
                test_acc[round_idx*self.args.update_round+epoch], \
                test_loss[round_idx*self.args.update_round+epoch], \
                train_rule_contr, test_rule_contr = self.eval_impl(epoch)
            rule_contr = train_rule_contr
        return train_acc, train_loss, test_acc, test_loss

    def train_impl(self, epoch_idx):
        self.model.train()
        for batch_idx, (x, labels) in enumerate(self.train_loader):
            # self.args.logger.info(images.shape)
            x, labels = x.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            log_probs, _ = self.model(x)
            loss = self.criterion(log_probs, labels.squeeze())
            loss.backward()
            self.optimizer.step()
            self.args.logger.info('Local Training Epoch: {} {}-th iters\t Loss: {:.6f}'.format(epoch_idx,
                                                                                               batch_idx, loss.item()))

    def eval_impl(self, epoch_idx):
        train_acc = 0.0
        train_loss = 0.0
        test_acc = 0.0
        test_loss = 0.0

        # train
        # if epoch_idx % self.args.frequency_of_train_acc_report == 0:
        train_acc, train_loss, train_rule_contr = self.test(b_is_train=True, epoch_idx=epoch_idx)

        # test
        # if epoch_idx % self.args.frequency_of_test_acc_report == 0:
        test_acc, test_loss, test_rule_contr = self.test(b_is_train=False, epoch_idx=epoch_idx)

        log_train_rule_contr = {f"train_rule{rule_idx + 1}": float(train_rule_contr[rule_idx]) for rule_idx in
                                torch.arange(train_rule_contr.shape[0])}
        log_test_rule_contr = {f"test_rule{rule_idx + 1}": float(test_rule_contr[rule_idx]) for rule_idx in
                               torch.arange(test_rule_contr.shape[0])}
        log_performance = {"epoch": epoch_idx, "Train Acc": train_acc, "Test Acc": test_acc,
                           "Train Loss": train_loss, "Test Loss": test_loss}
        log_report = {**log_train_rule_contr, **log_test_rule_contr, **log_performance}
        wandb.log(log_report)

        self.args.logger.debug(f"[epoch: {epoch_idx}, Train Acc: {train_acc}, Test Acc: {test_acc}, "
                               f"Train Loss: {train_loss}, Test Loss: {test_loss}")
        self.args.logger.debug(f"Train FS: {train_rule_contr.cpu()}")
        self.args.logger.debug(f"Test FS: {test_rule_contr.cpu()}")
        return train_acc, train_loss, test_acc, test_loss, train_rule_contr, test_rule_contr

    def test(self, b_is_train, epoch_idx):
        self.model.eval()
        contr_fs_arr = torch.empty(0, self.args.n_rule).int().to(self.args.device)
        metrics = {
            'test_correct': 0,
            'test_loss': 0,
            'test_precision': 0,
            'test_recall': 0,
            'test_total': 0
        }
        if b_is_train:
            test_data = self.train_loader
        else:
            test_data = self.test_loader
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(self.device)
                target = target.squeeze().to(self.device)
                pred, fire_strength = self.model(x)
                loss = self.criterion(pred, target.squeeze())

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                # _, contr_fs_tmp = torch.max(fire_strength, -1)
                # contr_fs_tmp = fire_strength.mean(0)
                contr_fs_arr = torch.cat([contr_fs_arr, fire_strength], 0)
                # contr_fs = torch.cat([contr_fs, contr_fs_tmp], 0)

                metrics['test_correct'] += correct.item()
                metrics['test_loss'] += loss.item() * target.size(0)
                metrics['test_total'] += target.size(0)

        acc, loss = self.save_log(b_is_train=b_is_train, metrics=metrics)
        # rule_contr = nn.functional.one_hot(contr_fs).sum(0)
        # if rule_contr.shape[0] < self.args.n_rule:
        #     n_diff = self.args.n_rule - rule_contr.shape[0]
        #     arr_diff = torch.zeros(n_diff).long().to(self.args.device)
        #     rule_contr = torch.cat([rule_contr, arr_diff], 0)
        # rule_contr = rule_contr / rule_contr.sum()
        rule_contr = contr_fs_arr.mean(0)
        return acc, loss, rule_contr

    def save_log(self, b_is_train, metrics):
        prefix = 'Train' if b_is_train else 'Test'

        all_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }

        all_metrics['num_samples'].append(copy.deepcopy(metrics['test_total']))
        all_metrics['num_correct'].append(copy.deepcopy(metrics['test_correct']))
        all_metrics['losses'].append(copy.deepcopy(metrics['test_loss']))

        # performance on all clients
        acc = sum(all_metrics['num_correct']) / sum(all_metrics['num_samples'])
        loss = sum(all_metrics['losses']) / sum(all_metrics['num_samples'])

        stats = {prefix + "/Loss": loss, prefix + '_acc': acc, prefix + '_loss': loss}
        # wandb.log({prefix + "/Acc": acc, "epoch": epoch_idx})
        # wandb.log({prefix + "/Loss": loss, "epoch": epoch_idx})

        # self.args.logger.debug(stats)
        return acc, loss

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)