import torch.nn as nn
import torch as pt
import torch.nn.functional as F
import torch


class MLPBlock(nn.Module):
    def __init__(self, hidden_dims, dropout_p=0.0):
        super(MLPBlock, self).__init__()
        # self.num_ftrs = hidden_dims[-1]
        self.dropout_p = dropout_p
        self.layers = nn.ModuleList([])
        for i in range(len(hidden_dims) - 1):
            ip_dim = hidden_dims[i]
            op_dim = hidden_dims[i + 1]
            self.layers.append(nn.Linear(ip_dim, op_dim, bias=True))
        self.__init_net_weights__()

    def __init_net_weights__(self):

        for m in self.layers:
            m.weight.data.normal_(0.0, 0.1)
            m.bias.data.fill_(0.1)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)

            # Do not apply ReLU on the final layer
            if i < len(self.layers):
                x = F.relu(x)

            # if i < (len(self.layers) - 1):  # No dropout on output layer
            #     x = F.dropout(x, p=self.dropout_p, training=self.training)

        return x


class MyMLP(nn.Module):
    def __init__(self, base_model, hidden_dims, out_dim, n_classes, dropout_p):
        super(MyMLP, self).__init__()

        self.features = MLPBlock(hidden_dims, dropout_p=dropout_p)
        num_ftrs = hidden_dims[-1]

        # summary(self.features.to('cuda:0'), (3,32,32))
        # print("features:", self.features)
        # projection MLP
        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, out_dim)

        # last layer
        self.l3 = nn.Linear(out_dim, n_classes)

    def _get_basemodel(self, model_name):
        try:
            model = self.model_dict[model_name]
            # print("Feature extractor:", model_name)
            return model
        except:
            raise (
                "Invalid model name. Check the config file and pass one of: resnet18 or resnet50"
            )

    def forward(self, x):
        h = self.features(x)
        # print("h before:", h)
        # print("h size:", h.size())
        h = h.squeeze()
        # print("h after:", h)
        x = self.l1(h)
        x = F.relu(x)
        x = self.l2(x)

        y = self.l3(x)
        return h, x, y


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
        out = F.relu(self.fc1(data))
        out = out + torch.randn_like(out)*self.sig
        out = nn.functional.dropout(out, p=self.drop_rt)
        out = F.relu(self.fc2(out))
        out = out + torch.randn_like(out) * self.sig
        out = nn.functional.dropout(out, p=self.drop_rt)
        return self.fc3(out)
    

class Mlp421D(DnnUncertain):
    def __init__(self, input_shape, n_cls, device, gnia_sig, dropout):
        super(Mlp421D, self).__init__(input_shape, n_cls, device, gnia_sig, dropout)
        self.fc1 = pt.nn.Linear(input_shape, int(input_shape*512)).to(device)

        self.fc2 = pt.nn.Linear(int(input_shape*512), int(input_shape*64)).to(device)
        self.fc3 = pt.nn.Linear(int(input_shape*64), n_cls).to(device)

    def forward(self, data):
        out = F.relu(self.fc1(data))
        out = out + torch.randn_like(out) * self.sig
        out = nn.functional.dropout(out, p=self.drop_rt)
        out = F.relu(self.fc2(out))
        out = out + torch.randn_like(out) * self.sig
        out = nn.functional.dropout(out, p=self.drop_rt)
        return self.fc3(out)


class Mlp21(DnnUncertain):
    def __init__(self, input_shape, n_cls, device, gnia_sig, dropout):
        super(Mlp21, self).__init__(input_shape, n_cls, device, gnia_sig, dropout)
        self.fc1 = pt.nn.Linear(input_shape, int(input_shape/2)).to(device)

        self.fc2 = pt.nn.Linear(int(input_shape/2), n_cls).to(device)

    def forward(self, data):
        out = F.relu(self.fc1(data))
        out = out + torch.randn_like(out) * self.sig
        out = nn.functional.dropout(out, p=self.drop_rt)
        return self.fc2(out)
    
    
class Mlp21D(DnnUncertain):
    def __init__(self, input_shape, n_cls, device, gnia_sig, dropout):
        super(Mlp21D, self).__init__(input_shape, n_cls, device, gnia_sig, dropout)
        self.fc1 = pt.nn.Linear(input_shape, int(input_shape*512)).to(device)

        self.fc2 = pt.nn.Linear(int(input_shape*512), n_cls).to(device)

    def forward(self, data):
        out = F.relu(self.fc1(data))
        out = out + torch.randn_like(out) * self.sig
        out = nn.functional.dropout(out, p=self.drop_rt)
        return self.fc2(out)


class Mlp121(DnnUncertain):
    def __init__(self, input_shape, n_cls, device, gnia_sig, dropout):
        super(Mlp121, self).__init__(input_shape, n_cls, device, gnia_sig, dropout)
        self.fc1 = pt.nn.Linear(input_shape, 2*input_shape).to(device)

        self.fc2 = pt.nn.Linear(2*input_shape, input_shape).to(device)
        self.fc3 = pt.nn.Linear(input_shape, n_cls).to(device)

    def forward(self, data):
        out = F.relu(self.fc1(data))
        out = out + torch.randn_like(out) * self.sig
        out = nn.functional.dropout(out, p=self.drop_rt)
        out = F.relu(self.fc2(out))
        out = out + torch.randn_like(out) * self.sig
        out = nn.functional.dropout(out, p=self.drop_rt)
        return self.fc3(out)


class Mlp121D(DnnUncertain):
    def __init__(self, input_shape, n_cls, device, gnia_sig, dropout):
        super(Mlp121D, self).__init__(input_shape, n_cls, device, gnia_sig, dropout)
        self.fc1 = pt.nn.Linear(input_shape, 512*input_shape).to(device)

        self.fc2 = pt.nn.Linear(512*input_shape, 64*input_shape).to(device)
        self.fc3 = pt.nn.Linear(64*input_shape, n_cls).to(device)

    def forward(self, data):
        out = F.relu(self.fc1(data))
        out = out + torch.randn_like(out) * self.sig
        out = nn.functional.dropout(out, p=self.drop_rt)
        out = F.relu(self.fc2(out))
        out = out + torch.randn_like(out) * self.sig
        out = nn.functional.dropout(out, p=self.drop_rt)
        return self.fc3(out)
    
    
class Mlp212(DnnUncertain):
    def __init__(self, input_shape, n_cls, device, gnia_sig, dropout):
        super(Mlp212, self).__init__(input_shape, n_cls, device, gnia_sig, dropout)
        self.fc1 = pt.nn.Linear(input_shape, int(input_shape/2)).to(device)

        self.fc2 = pt.nn.Linear(int(input_shape/2), input_shape).to(device)
        self.fc3 = pt.nn.Linear(input_shape, n_cls).to(device)

    def forward(self, data):
        out = F.relu(self.fc1(data))
        out = out + torch.randn_like(out) * self.sig
        out = nn.functional.dropout(out, p=self.drop_rt)
        out = F.relu(self.fc2(out))
        out = out + torch.randn_like(out) * self.sig
        out = nn.functional.dropout(out, p=self.drop_rt)
        return self.fc3(out)


class Mlp212D(DnnUncertain):
    def __init__(self, input_shape, n_cls, device, gnia_sig, dropout):
        super(Mlp212D, self).__init__(input_shape, n_cls, device, gnia_sig, dropout)
        self.fc1 = pt.nn.Linear(input_shape, int(input_shape*64)).to(device)

        self.fc2 = pt.nn.Linear(int(input_shape*64), input_shape*512).to(device)
        self.fc3 = pt.nn.Linear(input_shape*512, n_cls).to(device)

    def forward(self, data):
        out = F.relu(self.fc1(data))
        out = out + torch.randn_like(out) * self.sig
        out = nn.functional.dropout(out, p=self.drop_rt)
        out = F.relu(self.fc2(out))
        out = out + torch.randn_like(out) * self.sig
        out = nn.functional.dropout(out, p=self.drop_rt)
        return self.fc3(out)
    
    
class Mlp42124(DnnUncertain):
    def __init__(self, input_shape, n_cls, device, gnia_sig, dropout):
        super(Mlp42124, self).__init__(input_shape, n_cls, device, gnia_sig, dropout)
        self.fc1 = pt.nn.Linear(input_shape, int(input_shape/2)).to(device)

        self.fc2 = pt.nn.Linear(int(input_shape/2), int(input_shape/4)).to(device)
        self.fc3 = pt.nn.Linear(int(input_shape/4), int(input_shape/2)).to(device)
        self.fc4 = pt.nn.Linear(int(input_shape/2), input_shape).to(device)
        self.fc5 = pt.nn.Linear(input_shape, n_cls).to(device)

    def forward(self, data):
        out = F.relu(self.fc1(data))
        out = out + torch.randn_like(out) * self.sig
        out = nn.functional.dropout(out, p=self.drop_rt)
        out = F.relu(self.fc2(out))
        out = out + torch.randn_like(out) * self.sig
        out = nn.functional.dropout(out, p=self.drop_rt)
        out = nn.functional.relu(F.relu(self.fc3(out)))
        out = out + torch.randn_like(out) * self.sig
        out = nn.functional.dropout(out, p=self.drop_rt)
        out = nn.functional.relu(F.relu(self.fc4(out)))
        out = out + torch.randn_like(out) * self.sig
        out = nn.functional.dropout(out, p=self.drop_rt)
        return self.fc5(out)


class Mlp42124D(DnnUncertain):
    def __init__(self, input_shape, n_cls, device, gnia_sig, dropout):
        super(Mlp42124D, self).__init__(input_shape, n_cls, device, gnia_sig, dropout)
        self.fc1 = pt.nn.Linear(input_shape, int(input_shape*512)).to(device)

        self.fc2 = pt.nn.Linear(int(input_shape*512), int(input_shape*128)).to(device)
        self.fc3 = pt.nn.Linear(int(input_shape*128), int(input_shape*32)).to(device)
        self.fc4 = pt.nn.Linear(int(input_shape*32), input_shape*1).to(device)
        self.fc5 = pt.nn.Linear(input_shape*1, n_cls).to(device)

    def forward(self, data):
        out = F.relu(self.fc1(data))
        out = out + torch.randn_like(out) * self.sig
        out = nn.functional.dropout(out, p=self.drop_rt)
        out = F.relu(self.fc2(out))
        out = out + torch.randn_like(out) * self.sig
        out = nn.functional.dropout(out, p=self.drop_rt)
        out = nn.functional.relu(F.relu(self.fc3(out)))
        out = out + torch.randn_like(out) * self.sig
        out = nn.functional.dropout(out, p=self.drop_rt)
        out = nn.functional.relu(F.relu(self.fc4(out)))
        out = out + torch.randn_like(out) * self.sig
        out = nn.functional.dropout(out, p=self.drop_rt)
        return self.fc5(out)
    
    
class Mlp12421(DnnUncertain):
    def __init__(self, input_shape, n_cls, device, gnia_sig, dropout):
        super(Mlp12421, self).__init__(input_shape, n_cls, device, gnia_sig, dropout)
        self.fc1 = pt.nn.Linear(input_shape, 2*input_shape).to(device)

        self.fc2 = pt.nn.Linear(2*input_shape, 4*input_shape).to(device)
        self.fc3 = pt.nn.Linear(4 * input_shape, 2*input_shape).to(device)
        self.fc4 = pt.nn.Linear(2 * input_shape, input_shape).to(device)
        self.fc5 = pt.nn.Linear(input_shape, n_cls).to(device)

    def forward(self, data):
        out = F.relu(self.fc1(data))
        out = out + torch.randn_like(out) * self.sig
        out = nn.functional.dropout(out, p=self.drop_rt)
        out = F.relu(self.fc2(out))
        out = out + torch.randn_like(out) * self.sig
        out = nn.functional.dropout(out, p=self.drop_rt)
        out = nn.functional.relu(F.relu(self.fc3(out)))
        out = out + torch.randn_like(out) * self.sig
        out = nn.functional.dropout(out, p=self.drop_rt)
        out = nn.functional.relu(F.relu(self.fc4(out)))
        out = out + torch.randn_like(out) * self.sig
        out = nn.functional.dropout(out, p=self.drop_rt)
        return self.fc5(out)
    

class Mlp12421D(DnnUncertain):
    def __init__(self, input_shape, n_cls, device, gnia_sig, dropout):
        super(Mlp12421D, self).__init__(input_shape, n_cls, device, gnia_sig, dropout)
        self.fc1 = pt.nn.Linear(input_shape, 512).to(device)

        self.fc2 = pt.nn.Linear(512, 512*input_shape).to(device)
        self.fc3 = pt.nn.Linear(512 * input_shape, 128 * input_shape).to(device)
        self.fc4 = pt.nn.Linear(128 * input_shape, input_shape).to(device)
        self.fc5 = pt.nn.Linear(input_shape, n_cls).to(device)

    def forward(self, data):
        out = F.relu(self.fc1(data))
        out = out + torch.randn_like(out) * self.sig
        out = nn.functional.dropout(out, p=self.drop_rt)
        out = F.relu(self.fc2(out))
        out = out + torch.randn_like(out) * self.sig
        out = nn.functional.dropout(out, p=self.drop_rt)
        out = nn.functional.relu(F.relu(self.fc3(out)))
        out = out + torch.randn_like(out) * self.sig
        out = nn.functional.dropout(out, p=self.drop_rt)
        out = nn.functional.relu(F.relu(self.fc4(out)))
        out = out + torch.randn_like(out) * self.sig
        out = nn.functional.dropout(out, p=self.drop_rt)
        return self.fc5(out)
    
    
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
        out = self.layer1(data.unsqueeze(1))
        out = out + torch.randn_like(out) * self.sig
        out = nn.functional.dropout(out, p=self.drop_rt)
        out = self.layer2(out)
        out = out + torch.randn_like(out) * self.sig
        out = nn.functional.dropout(out, p=self.drop_rt)
        out = out.view(out.shape[0], -1)
        out = nn.functional.relu(self.fc1(out))
        out = out + torch.randn_like(out) * self.sig
        out = nn.functional.dropout(out, p=self.drop_rt)
        out = self.fc2(out)
        return out
    

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
        out = self.layer1(data.unsqueeze(1))
        out = out + torch.randn_like(out) * self.sig
        out = nn.functional.dropout(out, p=self.drop_rt)
        out = self.layer2(out)
        out = out + torch.randn_like(out) * self.sig
        out = nn.functional.dropout(out, p=self.drop_rt)
        out = out.view(out.shape[0], -1)
        out = nn.functional.relu(self.fc1(out))
        out = out + torch.randn_like(out) * self.sig
        out = nn.functional.dropout(out, p=self.drop_rt)
        out = self.fc2(out)
        return out


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
        out = self.layer1(data.unsqueeze(1))
        out = out + torch.randn_like(out) * self.sig
        out = nn.functional.dropout(out, p=self.drop_rt)
        out = self.layer2(out)
        out = out + torch.randn_like(out) * self.sig
        out = nn.functional.dropout(out, p=self.drop_rt)
        out = out.view(out.shape[0], -1)
        out = self.fc1(out)
        return out


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
        out = self.layer1(data.unsqueeze(1))
        out = out + torch.randn_like(out) * self.sig
        out = nn.functional.dropout(out, p=self.drop_rt)
        out = self.layer2(out)
        out = out + torch.randn_like(out) * self.sig
        out = nn.functional.dropout(out, p=self.drop_rt)
        out = out.view(out.shape[0], -1)
        out = self.fc1(out)
        return out
    
    
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
        out = self.layer1(data.unsqueeze(1))
        out = out + torch.randn_like(out) * self.sig
        out = nn.functional.dropout(out, p=self.drop_rt)
        out = out.view(out.shape[0], -1)
        out = self.fc1(out)
        return out


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
        out = self.layer1(data.unsqueeze(1))
        out = out + torch.randn_like(out) * self.sig
        out = nn.functional.dropout(out, p=self.drop_rt)
        out = out.view(out.shape[0], -1)
        out = self.fc1(out)
        return out
    
    
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
        out = self.layer1(data.unsqueeze(1))
        out = out + torch.randn_like(out) * self.sig
        out = nn.functional.dropout(out, p=self.drop_rt)
        out = out.view(out.shape[0], -1)
        out = nn.functional.relu(self.fc1(out))
        out = out + torch.randn_like(out) * self.sig
        out = nn.functional.dropout(out, p=self.drop_rt)
        out = nn.functional.relu(self.fc2(out))
        return out


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
        out = self.layer1(data.unsqueeze(1))
        out = out + torch.randn_like(out) * self.sig
        out = nn.functional.dropout(out, p=self.drop_rt)
        out = out.view(out.shape[0], -1)
        out = nn.functional.relu(self.fc1(out))
        out = out + torch.randn_like(out) * self.sig
        out = nn.functional.dropout(out, p=self.drop_rt)
        out = nn.functional.relu(self.fc2(out))
        return out
