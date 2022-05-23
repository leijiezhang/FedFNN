from abc import ABC, abstractmethod
from models.losses import BalancedSoftmax
import torch


class ModelTrainer(ABC):
    """Abstract base class for federated learning trainer.
       1. The goal of this abstract class is to be compatible to
       any deep learning frameworks such as PyTorch, TensorFlow, Keras, MXNET, etc.
       2. This class can be used in both server and client side
       3. This class is an operator which does not cache any states inside.
    """
    def __init__(self, model, args=None):
        self.model = model
        self.id = 0
        self.args = args

    def set_id(self, trainer_id):
        self.id = trainer_id

    @abstractmethod
    def get_model_params(self):
        pass

    @abstractmethod
    def set_model_params(self, model_parameters):
        pass

    @abstractmethod
    def train(self, train_data, device, args=None):
        pass

    @abstractmethod
    def test(self, test_data, device, args=None):
        pass

    @abstractmethod
    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        pass


class MyModelTrainer(ModelTrainer):
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args=None):
        model = self.model

        model.to(device)
        model.train()

        # train and update
        criterion = torch.nn.CrossEntropyLoss().to(device)
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
        for epoch in range(args.epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                model.zero_grad()
                log_probs, _ = model(x)
                loss = criterion(log_probs, labels.squeeze())
                loss.backward()

                # Uncommet this following line to avoid nan loss
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            args.logger.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(
                self.id, epoch, sum(epoch_loss) / len(epoch_loss)))

    def test(self, test_data, device, args=None):
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
        criterion = torch.nn.CrossEntropyLoss().to(device)
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

                fs = torch.cat([fs, fire_strength], 0)

                metrics_normal['test_correct'] += correct.item()
                metrics_normal['test_loss'] += loss.item() * target.size(0)
                metrics_normal['test_total'] += target.size(0)

        metrics_normal['fs'] = fs
        return metrics_normal

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False

