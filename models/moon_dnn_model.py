import torch.nn as nn
import torch as pt
import torch.nn.functional as F
import torch


class DnnUncertain(pt.nn.Module):
    def __init__(self, input_shape, n_cls, device, gnia_sig, dropout):
        super(DnnUncertain, self).__init__()
        self.input_shape = input_shape
        self.n_cls = n_cls
        self.device = device
        self.sig = gnia_sig
        self.drop_rt = dropout

    def forward(self, data):
        return data


class Mlp421(DnnUncertain):
    def __init__(self, input_shape, n_cls, device, gnia_sig, dropout):
        super(Mlp421, self).__init__(input_shape, n_cls, device, gnia_sig, dropout)
        self.fc1 = pt.nn.Linear(input_shape, int(input_shape/2)).to(device)

        self.fc2 = pt.nn.Linear(int(input_shape/2), int(input_shape/4)).to(device)
        self.fc3 = pt.nn.Linear(int(input_shape/4), n_cls).to(device)

    def forward(self, data):
        h = F.relu(self.fc1(data))
        h = h + torch.randn_like(h)*self.sig
        h = nn.functional.dropout(h, p=self.drop_rt)
        h = F.relu(self.fc2(h))
        h = h + torch.randn_like(h) * self.sig
        x = nn.functional.dropout(h, p=self.drop_rt)
        y = self.fc3(x)
        return h, x, y
    

class Mlp421D(DnnUncertain):
    def __init__(self, input_shape, n_cls, device, gnia_sig, dropout):
        super(Mlp421D, self).__init__(input_shape, n_cls, device, gnia_sig, dropout)
        self.fc1 = pt.nn.Linear(input_shape, int(input_shape*512)).to(device)

        self.fc2 = pt.nn.Linear(int(input_shape*512), int(input_shape*64)).to(device)
        self.fc3 = pt.nn.Linear(int(input_shape*64), n_cls).to(device)

    def forward(self, data):
        h = F.relu(self.fc1(data))
        h = h + torch.randn_like(h) * self.sig
        h = nn.functional.dropout(h, p=self.drop_rt)
        h = F.relu(self.fc2(h))
        h = h + torch.randn_like(h) * self.sig
        x = nn.functional.dropout(h, p=self.drop_rt)
        y = self.fc3(x)
        return h, x, y


class Mlp421D1(DnnUncertain):
    def __init__(self, input_shape, n_cls, device, gnia_sig, dropout):
        super(Mlp421D1, self).__init__(input_shape, n_cls, device, gnia_sig, dropout)
        self.fc1 = pt.nn.Linear(input_shape, int(input_shape*512)).to(device)

        self.fc2 = pt.nn.Linear(int(512), int(64)).to(device)
        self.fc3 = pt.nn.Linear(int(64), n_cls).to(device)

    def forward(self, data):
        h = F.relu(self.fc1(data))
        h = h + torch.randn_like(h) * self.sig
        h = nn.functional.dropout(h, p=self.drop_rt)
        h = F.relu(self.fc2(h))
        h = h + torch.randn_like(h) * self.sig
        x = nn.functional.dropout(h, p=self.drop_rt)
        y = self.fc3(x)
        return h, x, y


class Mlp21(DnnUncertain):
    def __init__(self, input_shape, n_cls, device, gnia_sig, dropout):
        super(Mlp21, self).__init__(input_shape, n_cls, device, gnia_sig, dropout)
        self.fc1 = pt.nn.Linear(input_shape, int(input_shape/2)).to(device)

        self.fc2 = pt.nn.Linear(int(input_shape/2), n_cls).to(device)

    def forward(self, data):
        h = F.relu(self.fc1(data))
        h = h + torch.randn_like(h) * self.sig
        x = nn.functional.dropout(h, p=self.drop_rt)
        y = self.fc2(x)
        return h, x, y
    
    
class Mlp21D(DnnUncertain):
    def __init__(self, input_shape, n_cls, device, gnia_sig, dropout):
        super(Mlp21D, self).__init__(input_shape, n_cls, device, gnia_sig, dropout)
        self.fc1 = pt.nn.Linear(input_shape, int(input_shape*512)).to(device)

        self.fc2 = pt.nn.Linear(int(input_shape*512), n_cls).to(device)

    def forward(self, data):
        h = F.relu(self.fc1(data))
        h = h + torch.randn_like(h) * self.sig
        x = nn.functional.dropout(h, p=self.drop_rt)
        y = self.fc2(x)
        return h, x, y


class Mlp21D1(DnnUncertain):
    def __init__(self, input_shape, n_cls, device, gnia_sig, dropout):
        super(Mlp21D1, self).__init__(input_shape, n_cls, device, gnia_sig, dropout)
        self.fc1 = pt.nn.Linear(input_shape, int(512)).to(device)

        self.fc2 = pt.nn.Linear(int(512), n_cls).to(device)

    def forward(self, data):
        h = F.relu(self.fc1(data))
        h = h + torch.randn_like(h) * self.sig
        x = nn.functional.dropout(h, p=self.drop_rt)
        y = self.fc2(x)
        return h, x, y


class Mlp121(DnnUncertain):
    def __init__(self, input_shape, n_cls, device, gnia_sig, dropout):
        super(Mlp121, self).__init__(input_shape, n_cls, device, gnia_sig, dropout)
        self.fc1 = pt.nn.Linear(input_shape, 2*input_shape).to(device)

        self.fc2 = pt.nn.Linear(2*input_shape, input_shape).to(device)
        self.fc3 = pt.nn.Linear(input_shape, n_cls).to(device)

    def forward(self, data):
        h = F.relu(self.fc1(data))
        h = h + torch.randn_like(h) * self.sig
        h = nn.functional.dropout(h, p=self.drop_rt)
        h = F.relu(self.fc2(h))
        h = h + torch.randn_like(h) * self.sig
        x = nn.functional.dropout(h, p=self.drop_rt)
        y = self.fc3(x)
        return h, x, y


class Mlp121D(DnnUncertain):
    def __init__(self, input_shape, n_cls, device, gnia_sig, dropout):
        super(Mlp121D, self).__init__(input_shape, n_cls, device, gnia_sig, dropout)
        self.fc1 = pt.nn.Linear(input_shape, 512*input_shape).to(device)

        self.fc2 = pt.nn.Linear(512*input_shape, 64*input_shape).to(device)
        self.fc3 = pt.nn.Linear(64*input_shape, n_cls).to(device)

    def forward(self, data):
        h = F.relu(self.fc1(data))
        h = h + torch.randn_like(h) * self.sig
        h = nn.functional.dropout(h, p=self.drop_rt)
        h = F.relu(self.fc2(h))
        h = h + torch.randn_like(h) * self.sig
        x = nn.functional.dropout(h, p=self.drop_rt)
        y = self.fc3(x)
        return h, x, y


class Mlp121D1(DnnUncertain):
    def __init__(self, input_shape, n_cls, device, gnia_sig, dropout):
        super(Mlp121D1, self).__init__(input_shape, n_cls, device, gnia_sig, dropout)
        self.fc1 = pt.nn.Linear(input_shape, 512).to(device)

        self.fc2 = pt.nn.Linear(512, 128).to(device)
        self.fc3 = pt.nn.Linear(128, n_cls).to(device)

    def forward(self, data):
        h = F.relu(self.fc1(data))
        h = h + torch.randn_like(h) * self.sig
        h = nn.functional.dropout(h, p=self.drop_rt)
        h = F.relu(self.fc2(h))
        h = h + torch.randn_like(h) * self.sig
        x = nn.functional.dropout(h, p=self.drop_rt)
        y = self.fc3(x)
        return h, x, y

    
class Mlp212(DnnUncertain):
    def __init__(self, input_shape, n_cls, device, gnia_sig, dropout):
        super(Mlp212, self).__init__(input_shape, n_cls, device, gnia_sig, dropout)
        self.fc1 = pt.nn.Linear(input_shape, int(input_shape/2)).to(device)

        self.fc2 = pt.nn.Linear(int(input_shape/2), input_shape).to(device)
        self.fc3 = pt.nn.Linear(input_shape, n_cls).to(device)

    def forward(self, data):
        h = F.relu(self.fc1(data))
        h = h + torch.randn_like(h) * self.sig
        h = nn.functional.dropout(h, p=self.drop_rt)
        h = F.relu(self.fc2(h))
        h = h + torch.randn_like(h) * self.sig
        x = nn.functional.dropout(h, p=self.drop_rt)
        y = self.fc3(x)
        return h, x, y


class Mlp212D(DnnUncertain):
    def __init__(self, input_shape, n_cls, device, gnia_sig, dropout):
        super(Mlp212D, self).__init__(input_shape, n_cls, device, gnia_sig, dropout)
        self.fc1 = pt.nn.Linear(input_shape, int(input_shape*64)).to(device)

        self.fc2 = pt.nn.Linear(int(input_shape*64), input_shape*512).to(device)
        self.fc3 = pt.nn.Linear(input_shape*512, n_cls).to(device)

    def forward(self, data):
        h = F.relu(self.fc1(data))
        h = h + torch.randn_like(h) * self.sig
        h = nn.functional.dropout(h, p=self.drop_rt)
        h = F.relu(self.fc2(h))
        h = h + torch.randn_like(h) * self.sig
        x = nn.functional.dropout(h, p=self.drop_rt)
        y = self.fc3(x)
        return h, x, y


class Mlp212D1(DnnUncertain):
    def __init__(self, input_shape, n_cls, device, gnia_sig, dropout):
        super(Mlp212D1, self).__init__(input_shape, n_cls, device, gnia_sig, dropout)
        self.fc1 = pt.nn.Linear(input_shape, int(64)).to(device)

        self.fc2 = pt.nn.Linear(int(64), 512).to(device)
        self.fc3 = pt.nn.Linear(512, n_cls).to(device)

    def forward(self, data):
        h = F.relu(self.fc1(data))
        h = h + torch.randn_like(h) * self.sig
        h = nn.functional.dropout(h, p=self.drop_rt)
        h = F.relu(self.fc2(h))
        h = h + torch.randn_like(h) * self.sig
        x = nn.functional.dropout(h, p=self.drop_rt)
        y = self.fc3(x)
        return h, x, y

    
class Mlp42124(DnnUncertain):
    def __init__(self, input_shape, n_cls, device, gnia_sig, dropout):
        super(Mlp42124, self).__init__(input_shape, n_cls, device, gnia_sig, dropout)
        self.fc1 = pt.nn.Linear(input_shape, int(input_shape/2)).to(device)

        self.fc2 = pt.nn.Linear(int(input_shape/2), int(input_shape/4)).to(device)
        self.fc3 = pt.nn.Linear(int(input_shape/4), int(input_shape/2)).to(device)
        self.fc4 = pt.nn.Linear(int(input_shape/2), input_shape).to(device)
        self.fc5 = pt.nn.Linear(input_shape, n_cls).to(device)

    def forward(self, data):
        h = F.relu(self.fc1(data))
        h = h + torch.randn_like(h) * self.sig
        h = nn.functional.dropout(h, p=self.drop_rt)
        h = F.relu(self.fc2(h))
        h = h + torch.randn_like(h) * self.sig
        h = nn.functional.dropout(h, p=self.drop_rt)
        h = nn.functional.relu(F.relu(self.fc3(h)))
        h = h + torch.randn_like(h) * self.sig
        h = nn.functional.dropout(h, p=self.drop_rt)
        x = nn.functional.relu(F.relu(self.fc4(h)))
        x = x + torch.randn_like(x) * self.sig
        x = nn.functional.dropout(x, p=self.drop_rt)
        y = self.fc5(x)
        return h, x, y


class Mlp42124D(DnnUncertain):
    def __init__(self, input_shape, n_cls, device, gnia_sig, dropout):
        super(Mlp42124D, self).__init__(input_shape, n_cls, device, gnia_sig, dropout)
        self.fc1 = pt.nn.Linear(input_shape, int(input_shape*512)).to(device)

        self.fc2 = pt.nn.Linear(int(input_shape*512), int(input_shape*128)).to(device)
        self.fc3 = pt.nn.Linear(int(input_shape*128), int(input_shape*32)).to(device)
        self.fc4 = pt.nn.Linear(int(input_shape*32), input_shape*1).to(device)
        self.fc5 = pt.nn.Linear(input_shape*1, n_cls).to(device)

    def forward(self, data):
        h = F.relu(self.fc1(data))
        h = h + torch.randn_like(h) * self.sig
        h = nn.functional.dropout(h, p=self.drop_rt)
        h = F.relu(self.fc2(h))
        h = h + torch.randn_like(h) * self.sig
        h = nn.functional.dropout(h, p=self.drop_rt)
        h = nn.functional.relu(F.relu(self.fc3(h)))
        h = h + torch.randn_like(h) * self.sig
        h = nn.functional.dropout(h, p=self.drop_rt)
        x = nn.functional.relu(F.relu(self.fc4(h)))
        x = x + torch.randn_like(x) * self.sig
        x = nn.functional.dropout(x, p=self.drop_rt)
        y = self.fc5(x)
        return h, x, y


class Mlp42124D1(DnnUncertain):
    def __init__(self, input_shape, n_cls, device, gnia_sig, dropout):
        super(Mlp42124D1, self).__init__(input_shape, n_cls, device, gnia_sig, dropout)
        self.fc1 = pt.nn.Linear(input_shape, int(512)).to(device)

        self.fc2 = pt.nn.Linear(int(512), int(256)).to(device)
        self.fc3 = pt.nn.Linear(int(256), int(128)).to(device)
        self.fc4 = pt.nn.Linear(int(128), 64).to(device)
        self.fc5 = pt.nn.Linear(64, n_cls).to(device)

    def forward(self, data):
        h = F.relu(self.fc1(data))
        h = h + torch.randn_like(h) * self.sig
        h = nn.functional.dropout(h, p=self.drop_rt)
        h = F.relu(self.fc2(h))
        h = h + torch.randn_like(h) * self.sig
        h = nn.functional.dropout(h, p=self.drop_rt)
        h = nn.functional.relu(F.relu(self.fc3(h)))
        h = h + torch.randn_like(h) * self.sig
        h = nn.functional.dropout(h, p=self.drop_rt)
        x = nn.functional.relu(F.relu(self.fc4(h)))
        x = x + torch.randn_like(x) * self.sig
        x = nn.functional.dropout(x, p=self.drop_rt)
        y = self.fc5(x)
        return h, x, y

    
class Mlp12421(DnnUncertain):
    def __init__(self, input_shape, n_cls, device, gnia_sig, dropout):
        super(Mlp12421, self).__init__(input_shape, n_cls, device, gnia_sig, dropout)
        self.fc1 = pt.nn.Linear(input_shape, 2*input_shape).to(device)

        self.fc2 = pt.nn.Linear(2*input_shape, 4*input_shape).to(device)
        self.fc3 = pt.nn.Linear(4 * input_shape, 2*input_shape).to(device)
        self.fc4 = pt.nn.Linear(2 * input_shape, input_shape).to(device)
        self.fc5 = pt.nn.Linear(input_shape, n_cls).to(device)

    def forward(self, data):
        h = F.relu(self.fc1(data))
        h = h + torch.randn_like(h) * self.sig
        h = nn.functional.dropout(h, p=self.drop_rt)
        h = F.relu(self.fc2(h))
        h = h + torch.randn_like(h) * self.sig
        h = nn.functional.dropout(h, p=self.drop_rt)
        h = nn.functional.relu(F.relu(self.fc3(h)))
        h = h + torch.randn_like(h) * self.sig
        h = nn.functional.dropout(h, p=self.drop_rt)
        x = nn.functional.relu(F.relu(self.fc4(h)))
        x = x + torch.randn_like(x) * self.sig
        x = nn.functional.dropout(x, p=self.drop_rt)
        y = self.fc5(x)
        return h, x, y
    

class Mlp12421D(DnnUncertain):
    def __init__(self, input_shape, n_cls, device, gnia_sig, dropout):
        super(Mlp12421D, self).__init__(input_shape, n_cls, device, gnia_sig, dropout)
        self.fc1 = pt.nn.Linear(input_shape, 512).to(device)

        self.fc2 = pt.nn.Linear(512, 512*input_shape).to(device)
        self.fc3 = pt.nn.Linear(512 * input_shape, 128 * input_shape).to(device)
        self.fc4 = pt.nn.Linear(128 * input_shape, input_shape).to(device)
        self.fc5 = pt.nn.Linear(input_shape, n_cls).to(device)

    def forward(self, data):
        h = F.relu(self.fc1(data))
        h = h + torch.randn_like(h) * self.sig
        h = nn.functional.dropout(h, p=self.drop_rt)
        h = F.relu(self.fc2(h))
        h = h + torch.randn_like(h) * self.sig
        h = nn.functional.dropout(h, p=self.drop_rt)
        h = nn.functional.relu(F.relu(self.fc3(h)))
        h = h + torch.randn_like(h) * self.sig
        h = nn.functional.dropout(h, p=self.drop_rt)
        x = nn.functional.relu(F.relu(self.fc4(h)))
        x = x + torch.randn_like(x) * self.sig
        x = nn.functional.dropout(x, p=self.drop_rt)
        y = self.fc5(x)
        return h, x, y


class Mlp12421D1(DnnUncertain):
    def __init__(self, input_shape, n_cls, device, gnia_sig, dropout):
        super(Mlp12421D1, self).__init__(input_shape, n_cls, device, gnia_sig, dropout)
        self.fc1 = pt.nn.Linear(input_shape, 512).to(device)

        self.fc2 = pt.nn.Linear(512, 512).to(device)
        self.fc3 = pt.nn.Linear(512, 128).to(device)
        self.fc4 = pt.nn.Linear(128, 64).to(device)
        self.fc5 = pt.nn.Linear(64, n_cls).to(device)

    def forward(self, data):
        h = F.relu(self.fc1(data))
        h = h + torch.randn_like(h) * self.sig
        h = nn.functional.dropout(h, p=self.drop_rt)
        h = F.relu(self.fc2(h))
        h = h + torch.randn_like(h) * self.sig
        h = nn.functional.dropout(h, p=self.drop_rt)
        h = nn.functional.relu(F.relu(self.fc3(h)))
        h = h + torch.randn_like(h) * self.sig
        h = nn.functional.dropout(h, p=self.drop_rt)
        x = nn.functional.relu(F.relu(self.fc4(h)))
        x = x + torch.randn_like(x) * self.sig
        x = nn.functional.dropout(x, p=self.drop_rt)
        y = self.fc5(x)
        return h, x, y


class Cnn32(DnnUncertain):
    def __init__(self, input_shape, n_cls, device, gnia_sig, dropout):
        super(Cnn32, self).__init__(input_shape, n_cls, device, gnia_sig, dropout)
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 3, kernel_size=tuple([3]), padding=1),
            nn.BatchNorm1d(3, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2)).to(device)
        self.layer2 = nn.Sequential(
            nn.Conv1d(3, 6, kernel_size=tuple([3]), padding=1),
            nn.BatchNorm1d(6, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2)).to(device)
        self.layer3 = nn.Sequential(
            nn.Conv1d(6, 9, kernel_size=tuple([3]), padding=1),
            nn.BatchNorm1d(9, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2)).to(device)
        input_size = int(((self.input_shape + 1 * 2 - 3) / 1 + 1) / 2)
        input_size = int(((input_size + 1 * 2 - 3) / 1 + 1) / 2)
        input_size = int(((input_size + 1 * 2 - 3) / 1 + 1) / 2)
        self.fc1 = nn.Linear(input_size*9, int(input_size*9/2)).to(device)
        self.fc2 = nn.Linear(int(input_size*9/2), self.n_cls).to(device)

    def forward(self, data):
        h = self.layer1(data.unsqueeze(1))
        h = h + torch.randn_like(h) * self.sig
        h = nn.functional.dropout(h, p=self.drop_rt)
        h = self.layer2(h)
        h = h + torch.randn_like(h) * self.sig
        h = nn.functional.dropout(h, p=self.drop_rt)
        h = self.layer3(h)
        h = h + torch.randn_like(h) * self.sig
        h = nn.functional.dropout(h, p=self.drop_rt)
        h = h.view(h.shape[0], -1)
        x = nn.functional.relu(self.fc1(h))
        x = x + torch.randn_like(x) * self.sig
        x = nn.functional.dropout(x, p=self.drop_rt)
        y = self.fc2(x)
        return h, x, y


class Cnn32D(DnnUncertain):
    def __init__(self, input_shape, n_cls, device, gnia_sig, dropout):
        super(Cnn32D, self).__init__(input_shape, n_cls, device, gnia_sig, dropout)
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, self.input_shape, kernel_size=tuple([3]), padding=1),
            nn.BatchNorm1d(self.input_shape, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2)).to(device)
        self.layer2 = nn.Sequential(
            nn.Conv1d(self.input_shape, 2*self.input_shape, kernel_size=tuple([3]), padding=1),
            nn.BatchNorm1d(2*self.input_shape, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2)).to(device)
        self.layer3 = nn.Sequential(
            nn.Conv1d(2*self.input_shape, 4*self.input_shape, kernel_size=tuple([3]), padding=1),
            nn.BatchNorm1d(4*self.input_shape, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2)).to(device)
        input_size = int(((self.input_shape + 1 * 2 - 3) / 1 + 1) / 2)
        input_size = int(((input_size + 1 * 2 - 3) / 1 + 1) / 2)
        input_size = int(((input_size + 1 * 2 - 3) / 1 + 1) / 2)
        self.fc1 = nn.Linear(input_size * 2 * self.input_shape, int(input_size * self.input_shape)).to(device)
        self.fc2 = nn.Linear(int(input_size * self.input_shape), self.n_cls).to(device)

    def forward(self, data):
        h = self.layer1(data.unsqueeze(1))
        h = h + torch.randn_like(h) * self.sig
        h = nn.functional.dropout(h, p=self.drop_rt)
        h = self.layer2(h)
        h = h + torch.randn_like(h) * self.sig
        h = nn.functional.dropout(h, p=self.drop_rt)
        h = self.layer3(h)
        h = h + torch.randn_like(h) * self.sig
        h = nn.functional.dropout(h, p=self.drop_rt)
        h = h.view(h.shape[0], -1)
        x = nn.functional.relu(self.fc1(h))
        x = x + torch.randn_like(x) * self.sig
        x = nn.functional.dropout(x, p=self.drop_rt)
        y = self.fc2(x)
        return h, x, y


class Cnn22(DnnUncertain):
    def __init__(self, input_shape, n_cls, device, gnia_sig, dropout):
        super(Cnn22, self).__init__(input_shape, n_cls, device, gnia_sig, dropout)
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 3, kernel_size=tuple([3]), padding=1),
            nn.BatchNorm1d(3, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2)).to(device)
        self.layer2 = nn.Sequential(
            nn.Conv1d(3, 6, kernel_size=tuple([3]), padding=1),
            nn.BatchNorm1d(6, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2)).to(device)
        input_size = int(((self.input_shape + 1 * 2 - 3) / 1 + 1) / 2)
        input_size = int(((input_size + 1 * 2 - 3) / 1 + 1) / 2)
        self.fc1 = nn.Linear(input_size*6, int(input_size*6/2)).to(device)
        self.fc2 = nn.Linear(int(input_size*6/2), self.n_cls).to(device)

    def forward(self, data):
        h = self.layer1(data.unsqueeze(1))
        h = h + torch.randn_like(h) * self.sig
        h = nn.functional.dropout(h, p=self.drop_rt)
        h = self.layer2(h)
        h = h + torch.randn_like(h) * self.sig
        h = nn.functional.dropout(h, p=self.drop_rt)
        h = h.view(h.shape[0], -1)
        x = nn.functional.relu(self.fc1(h))
        x = x + torch.randn_like(x) * self.sig
        x = nn.functional.dropout(x, p=self.drop_rt)
        y = self.fc2(x)
        return h, x, y
    

class Cnn22D(DnnUncertain):
    def __init__(self, input_shape, n_cls, device, gnia_sig, dropout):
        super(Cnn22D, self).__init__(input_shape, n_cls, device, gnia_sig, dropout)
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, self.input_shape, kernel_size=tuple([3]), padding=1),
            nn.BatchNorm1d(self.input_shape, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2)).to(device)
        self.layer2 = nn.Sequential(
            nn.Conv1d(self.input_shape, 2*self.input_shape, kernel_size=tuple([3]), padding=1),
            nn.BatchNorm1d(2*self.input_shape, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2)).to(device)
        input_size = int(((self.input_shape + 1 * 2 - 3) / 1 + 1) / 2)
        input_size = int(((input_size + 1 * 2 - 3) / 1 + 1) / 2)
        self.fc1 = nn.Linear(input_size*2*self.input_shape, int(input_size*self.input_shape)).to(device)
        self.fc2 = nn.Linear(int(input_size*self.input_shape), self.n_cls).to(device)

    def forward(self, data):
        h = self.layer1(data.unsqueeze(1))
        h = h + torch.randn_like(h) * self.sig
        h = nn.functional.dropout(h, p=self.drop_rt)
        h = self.layer2(h)
        h = h + torch.randn_like(h) * self.sig
        h = nn.functional.dropout(h, p=self.drop_rt)
        h = h.view(h.shape[0], -1)
        x = nn.functional.relu(self.fc1(h))
        x = x + torch.randn_like(x) * self.sig
        x = nn.functional.dropout(x, p=self.drop_rt)
        y = self.fc2(x)
        return h, x, y


class Cnn21(DnnUncertain):
    def __init__(self, input_shape, n_cls, device, gnia_sig, dropout):
        super(Cnn21, self).__init__(input_shape, n_cls, device, gnia_sig, dropout)
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 3, kernel_size=tuple([3]), padding=1),
            nn.BatchNorm1d(3, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2)).to(device)
        self.layer2 = nn.Sequential(
            nn.Conv1d(3, 6, kernel_size=tuple([3]), padding=1),
            nn.BatchNorm1d(6, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2)).to(device)
        input_size = int(((self.input_shape+1*2-3)/1 + 1)/2)
        input_size = int(((input_size+1*2-3)/1 + 1)/2)
        self.fc1 = nn.Linear(input_size*6, self.n_cls).to(device)

    def forward(self, data):
        h = self.layer1(data.unsqueeze(1))
        h = h + torch.randn_like(h) * self.sig
        h = nn.functional.dropout(h, p=self.drop_rt)
        x = self.layer2(h)
        x = x + torch.randn_like(x) * self.sig
        x = nn.functional.dropout(x, p=self.drop_rt)
        x = x.view(x.shape[0], -1)
        y = self.fc1(x)
        return h, x, y


class Cnn21D(DnnUncertain):
    def __init__(self, input_shape, n_cls, device, gnia_sig, dropout):
        super(Cnn21D, self).__init__(input_shape, n_cls, device, gnia_sig, dropout)
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, self.input_shape, kernel_size=tuple([3]), padding=1),
            nn.BatchNorm1d(self.input_shape, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2)).to(device)
        self.layer2 = nn.Sequential(
            nn.Conv1d(self.input_shape, 2*self.input_shape, kernel_size=tuple([3]), padding=1),
            nn.BatchNorm1d(2*self.input_shape, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2)).to(device)
        input_size = int(((self.input_shape+1*2-3)/1 + 1)/2)
        input_size = int(((input_size+1*2-3)/1 + 1)/2)
        self.fc1 = nn.Linear(input_size*2*self.input_shape, self.n_cls).to(device)

    def forward(self, data):
        h = self.layer1(data.unsqueeze(1))
        h = h + torch.randn_like(h) * self.sig
        h = nn.functional.dropout(h, p=self.drop_rt)
        x = self.layer2(h)
        x = x + torch.randn_like(x) * self.sig
        x = nn.functional.dropout(x, p=self.drop_rt)
        x = x.view(x.shape[0], -1)
        y = self.fc1(x)
        return h, x, y
    
    
class Cnn11(DnnUncertain):
    def __init__(self, input_shape, n_cls, device, gnia_sig, dropout):
        super(Cnn11, self).__init__(input_shape, n_cls, device, gnia_sig, dropout)
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 3, kernel_size=tuple([3]), padding=1),
            nn.BatchNorm1d(3, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2)).to(device)
        input_size = int(((self.input_shape+1*2-3)/1 + 1)/2)
        self.fc1 = nn.Linear(input_size*3, self.n_cls).to(device)

    def forward(self, data):
        h = self.layer1(data.unsqueeze(1))
        h = h + torch.randn_like(h) * self.sig
        h = nn.functional.dropout(h, p=self.drop_rt)
        h = h.view(h.shape[0], -1)
        y = self.fc1(h)
        return h, h, y


class Cnn11D(DnnUncertain):
    def __init__(self, input_shape, n_cls, device, gnia_sig, dropout):
        super(Cnn11D, self).__init__(input_shape, n_cls, device, gnia_sig, dropout)
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, self.input_shape, kernel_size=tuple([3]), padding=1),
            nn.BatchNorm1d(self.input_shape, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2)).to(device)
        input_size = int(((self.input_shape+1*2-3)/1 + 1)/2)
        self.fc1 = nn.Linear(input_size*self.input_shape, self.n_cls).to(device)

    def forward(self, data):
        h = self.layer1(data.unsqueeze(1))
        h = h + torch.randn_like(h) * self.sig
        h = nn.functional.dropout(h, p=self.drop_rt)
        h = h.view(h.shape[0], -1)
        y = self.fc1(h)
        return h, h, y
    
    
class Cnn12(DnnUncertain):
    def __init__(self, input_shape, n_cls, device, gnia_sig, dropout):
        super(Cnn12, self).__init__(input_shape, n_cls, device, gnia_sig, dropout)
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 3, kernel_size=tuple([3]), padding=1),
            nn.BatchNorm1d(3, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2)).to(device)
        input_size = int(((self.input_shape+1*2-3)/1 + 1)/2)
        self.fc1 = nn.Linear(input_size * 3, int(input_size * 3 / 2)).to(device)
        self.fc2 = nn.Linear(int(input_size * 3 / 2), self.n_cls).to(device)

    def forward(self, data):
        h = self.layer1(data.unsqueeze(1))
        h = h + torch.randn_like(h) * self.sig
        h = nn.functional.dropout(h, p=self.drop_rt)
        h = h.view(h.shape[0], -1)
        x = nn.functional.relu(self.fc1(h))
        x = x + torch.randn_like(x) * self.sig
        x = nn.functional.dropout(x, p=self.drop_rt)
        y = nn.functional.relu(self.fc2(x))
        return h, x, y


class Cnn12D(DnnUncertain):
    def __init__(self, input_shape, n_cls, device, gnia_sig, dropout):
        super(Cnn12D, self).__init__(input_shape, n_cls, device, gnia_sig, dropout)
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, self.input_shape, kernel_size=tuple([3]), padding=1),
            nn.BatchNorm1d(self.input_shape, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2)).to(device)
        input_size = int(((self.input_shape+1*2-3)/1 + 1)/2)
        self.fc1 = nn.Linear(input_size * self.input_shape, int(input_size * self.input_shape / 2)).to(device)
        self.fc2 = nn.Linear(int(input_size * self.input_shape / 2), self.n_cls).to(device)

    def forward(self, data):
        h = self.layer1(data.unsqueeze(1))
        h = h + torch.randn_like(h) * self.sig
        h = nn.functional.dropout(h, p=self.drop_rt)
        h = h.view(h.shape[0], -1)
        x = nn.functional.relu(self.fc1(h))
        x = x + torch.randn_like(x) * self.sig
        x = nn.functional.dropout(x, p=self.drop_rt)
        y = nn.functional.relu(self.fc2(x))
        return h, x, y
