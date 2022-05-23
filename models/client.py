from models.losses import BalancedSoftmax
import torch


class Client:

    def __init__(self, client_idx, local_training_data, local_test_data, local_sample_number, local_class_count,
                 args, device, model_trainer):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        self.local_class_count = local_class_count

        self.args = args
        self.args.logger.info(f"client{client_idx} sample number {self.local_sample_number}=> {self.local_class_count}")
        self.device = device
        self.model_trainer = model_trainer

    def update_local_dataset(self, client_idx, local_training_data, local_test_data, local_sample_number,
                             local_class_count):
        self.client_idx = client_idx
        self.model_trainer.set_id(client_idx)
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        self.local_class_count = local_class_count

    def get_sample_number(self):
        return self.local_sample_number

    def get_class_count(self):
        return self.local_class_count

    def train(self, w_global):
        self.model_trainer.set_model_params(w_global)
        self.model_trainer.train(self.local_training_data, self.device, self.args)
        weights = self.model_trainer.get_model_params()
        return weights

    def local_test(self, b_use_test_dataset):
        if b_use_test_dataset:
            test_data = self.local_test_data
        else:
            test_data = self.local_training_data
        metrics = self.model_trainer.test(test_data, self.device, self.args)
        return metrics


class FedFPNNClient:

    def __init__(self, client_idx, rules_idx_list, local_training_data, local_test_data, local_sample_number,
                 local_class_count,
                 args, model):
        self.client_idx = client_idx
        self.rules_idx_list = rules_idx_list
        self.n_rule_limit = args.n_rule_min
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        self.local_class_count = local_class_count

        self.args = args
        self.args.logger.info(f"client{client_idx} sample number {self.local_sample_number}=> {self.local_class_count}")
        self.device = args.device
        self.model = model

    def update_local_dataset(self, client_idx, local_training_data, local_test_data, local_sample_number,
                             local_class_count):
        self.client_idx = client_idx
        # self.model_trainer.set_id(client_idx)
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        self.local_class_count = local_class_count

    def update_rule_idx_list(self, rules_idx_list):
        self.rules_idx_list = rules_idx_list

    def get_sample_number(self):
        return self.local_sample_number

    def get_class_count(self):
        return self.local_class_count

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def _train_impl(self, train_data, device):
        model = self.model

        model.to(device)
        model.train()

        # train and update
        # criterion = torch.nn.CrossEntropyLoss().to(device)
        if self.args.criterion == "ce":
            criterion = torch.nn.CrossEntropyLoss().to(device)
        elif self.args.criterion == "bce":
            criterion = BalancedSoftmax().to(device)
        else:
            criterion = BalancedSoftmax().to(device)

        if self.args.optimizer == "sgd":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.lr)
        elif self.args.optimizer == "adam":
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.lr,
                                         weight_decay=self.args.wd, amsgrad=True)
        else:
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.lr)

        epoch_loss = []
        for epoch in range(self.args.epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                model.zero_grad()
                log_probs, fs_tmp = model(x)
                loss = criterion(log_probs, labels.squeeze())
                loss.backward()

                # Uncommet this following line to avoid nan loss
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            self.args.logger.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(
                self.client_idx, epoch, sum(epoch_loss) / len(epoch_loss)))

    def _test_impl(self, test_data, device):
        model = self.model

        model.to(device)
        model.eval()

        metrics_normal = {
            'test_correct': 0,
            'test_loss': 0,
            'test_total': 0,
            'rule_cndt': 0
        }

        fs = torch.empty(0, self.args.n_rule).to(self.args.device)
        # criterion = torch.nn.CrossEntropyLoss().to(device)
        if self.args.criterion == "ce":
            criterion = torch.nn.CrossEntropyLoss().to(device)
        elif self.args.criterion == "bce":
            criterion = BalancedSoftmax().to(device)
        else:
            criterion = BalancedSoftmax().to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
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

    def train(self, w_global):
        self.model.update_rules_idx_list(self.rules_idx_list)
        self.model.update_model(w_global)
        self._train_impl(self.local_training_data, self.device)
        weights = self.get_model_params()
        return weights

    def local_test(self, b_use_test_dataset):
        self.model.update_rules_idx_list(self.rules_idx_list)
        if b_use_test_dataset:
            test_data = self.local_test_data
        else:
            test_data = self.local_training_data
        metrics = self._test_impl(test_data, self.device)
        return metrics

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False
