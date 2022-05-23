import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from collections import OrderedDict


class AntecedentLayerR(nn.Module):
    """
    This is the antecedent layer of FPNN based on BP, the prototype and variance are initialized randomly
    """

    def __init__(self, n_fea):
        """
        :param n_fea: feature number of samples
        """
        super(AntecedentLayerR, self).__init__()
        # parameters in network
        prototype = torch.randn(1, n_fea)
        variance = torch.randn(1, n_fea)
        self.proto = nn.Parameter(prototype, requires_grad=True)
        self.var = nn.Parameter(variance, requires_grad=True)

        self.var_process = nn.Sequential(
            # nn.ReLU(),
            nn.Threshold(1, 1)
        )

        self.proto_process = nn.Sequential(
            # nn.ReLU(),
            nn.Threshold(1, 1)
        )

    def forward(self, data: torch.Tensor):
        # membership_values = torch.exp(-(data - self.proto) ** 2 * (2 * self.var_process(self.var) ** 2))
        # membership_values = torch.exp(-(data - torch.clamp(self.proto, -1, 1)) ** 2 / (2 * torch.clamp(
        #     self.var, 1e-4, 1) ** 2))
        membership_values = torch.exp(-(data - self.proto) ** 2 / (2 * torch.clamp(
            self.var, 1e-4, 1e-1) ** 2))

        return membership_values


class AntecedentLayerC(nn.Module):
    """
    This is the antecedent layer of FPNN based on BP, the prototype and variance are initialized by Federated K-means
    """

    def __init__(self, prototype: torch.Tensor, variance: torch.Tensor):
        """
        :param prototype: the center of this rule
        :param variance: the variance of this rule
        """
        super(AntecedentLayerC, self).__init__()
        # parameters in network
        self.proto = nn.Parameter(prototype, requires_grad=True)
        self.var = nn.Parameter(variance, requires_grad=True)

        self.var_process = nn.Sequential(
            # nn.ReLU(),
            nn.Threshold(0, 1)
        )

    def forward(self, data: torch.Tensor):
        membership_values = torch.exp(-(data - self.proto) ** 2 * (2 * self.var_process(self.var) ** 2))

        return membership_values


class ConsequentLayer(nn.Module):
    """
    This is the consequent layer of FPNN based on BP
    """

    def __init__(self, n_fea, num_class):
        """
        :param n_fea: feature number of samples
        :param num_class: the number of categories
        """
        super(ConsequentLayer, self).__init__()
        # parameters in network
        self.consq_layers = nn.Linear(n_fea, num_class)

    def forward(self, data: torch.Tensor):
        output = self.consq_layers(data)

        return output


class FSLayer(nn.Module):
    """
    This is the firing strength layer of FPNN based on BP
    """

    def __init__(self, n_fea, dropout_rate):
        """
        :param n_fea: feature number of samples
        """
        super(FSLayer, self).__init__()
        # parameters in network
        self.fs_layers = nn.Sequential(
            nn.Linear(n_fea, 2 * n_fea),
            nn.ELU(),
            nn.Linear(2 * n_fea, n_fea),
            nn.ELU(),
            nn.Linear(n_fea, 1),
            # nn.Tanh()
            nn.Dropout(p=dropout_rate)
        )

    def forward(self, data: torch.Tensor):
        output = self.fs_layers(data)
        return output


class FSLayerSum(nn.Module):
    """
    This is the firing strength layer of FPNN based on BP
    """

    def __init__(self):
        """
        :param n_fea: feature number of samples
        """
        super(FSLayerSum, self).__init__()
        # parameters in network

    def forward(self, data: torch.Tensor):
        output = data.sum(1).unsqueeze(1)
        return output
    

class FSLayerL2(nn.Module):
    """
    This is the firing strength layer of FPNN based on BP
    """

    def __init__(self):
        """
        :param n_fea: feature number of samples
        """
        super(FSLayerL2, self).__init__()
        # parameters in network

    def forward(self, data: torch.Tensor):
        output = (data*data).sum(1).unsqueeze(1)
        return output


class FSLayerProduct(nn.Module):
    """
    This is the firing strength layer of FPNN based on BP
    """

    def __init__(self):
        """
        :param n_fea: feature number of samples
        """
        super(FSLayerProduct, self).__init__()
        # parameters in network

    def forward(self, data: torch.Tensor):
        output = data.prod(1).unsqueeze(1)
        return output


class RuleR(nn.Module):
    """
    This is the FPNN based on BP, the prototype and variance are initialized randomly
    """

    def __init__(self, n_fea, num_class):
        """
        :param n_fea: feature number of samples
        :param num_class: the number of categories
        """
        super(RuleR, self).__init__()
        self.antecedent_layer = AntecedentLayerR(n_fea)
        self.consequent_layer = ConsequentLayer(n_fea, num_class)


class RuleC(nn.Module):
    """
    This is the FPNN based on BP, the prototype and variance are initialized randomly
    """

    def __init__(self, prototype: torch.Tensor, variance: torch.Tensor, n_fea, num_class, client_idx_list):
        """
        :param p_args: parameter list
        """
        super(RuleC, self).__init__()
        self.antecedent_layer = AntecedentLayerC(prototype, variance)
        self.consequent_layer = ConsequentLayer(n_fea, num_class)
        self.client_idx_list = client_idx_list


class FedFPNNR(nn.Module):
    """
    This is the FPNN based on BP, the prototype and variance are initialized randomly
    """

    def __init__(self, p_args, rules_idx_list):
        """
        :param n_fea: feature number of samples
        :param num_class: the number of categories
        """
        super(FedFPNNR, self).__init__()
        self.args = p_args
        self.rules_idx_list = rules_idx_list
        self.fs_layer = FSLayer(p_args.n_fea, p_args.dropout).to(p_args.device)
        if p_args.fs == 'sum':
            self.fs_layer = FSLayerSum().to(p_args.device)
        elif p_args.fs == 'prod':
            self.fs_layer = FSLayerProduct().to(p_args.device)
        self.rule_list = [RuleR(p_args.n_fea, p_args.n_class).to(p_args.device) for _ in range(p_args.n_rule)]
        for i, rules_item in enumerate(self.rule_list):
            self.add_module(f"rule_{i}", rules_item)

    def update_model(self, w_update):
        # # select local parameters from the global parameter
        # w_local = OrderedDict({})
        # for key, val in w_global.items():
        #     if "fs_layers" in key:
        #         w_local[key] = w_global[key]
        # for rule_idx in self.local_rule_idxs:
        #     for key, val in w_global.items():
        #         if f"local_rule_{rule_idx}" in key:
        #             w_local[key] = w_global[key]
        self.load_state_dict(w_update)

    def update_rules_idx_list(self, rules_idx_list):
        self.rules_idx_list = rules_idx_list

    def forward(self, data: torch.Tensor):
        n_batch = data.shape[0]
        # activate prototypes
        # produce antecedent layer
        fuzzy_set = torch.cat([rules_item.antecedent_layer(data).unsqueeze(0) for rules_item
                               in self.rule_list], dim=0)

        fire_strength_ini = torch.cat([self.fs_layer(data_diff_item) for data_diff_item in fuzzy_set], dim=1)
        fire_strength = F.softmax(fire_strength_ini[:, self.rules_idx_list], dim=1)

        # produce consequent layer
        data_processed = torch.cat([F.relu(rules_item.consequent_layer(data)).unsqueeze(0) for rules_item
                                    in self.rule_list], dim=0)
        data_processed = data_processed.view(self.args.n_rule, n_batch, self.args.n_class)

        outputs = torch.cat([(data_processed_item * fire_strength_item.unsqueeze(1)).unsqueeze(0)
                             for data_processed_item, fire_strength_item in
                             zip(data_processed[self.rules_idx_list], fire_strength.t())],
                            dim=0).sum(0)

        return outputs, fire_strength


class FPNN(nn.Module):
    """
    This is the FPNN based on BP, the prototype and variance are initialized randomly
    """

    def __init__(self, p_args, rules_idx_list):
        """
        :param n_fea: feature number of samples
        :param num_class: the number of categories
        """
        super(FPNN, self).__init__()
        self.args = p_args
        self.rules_idx_list = rules_idx_list
        self.fs_layer = FSLayer(p_args.n_fea, p_args.dropout).to(p_args.device)
        if p_args.fs == 'sum':
            self.fs_layer = FSLayerSum().to(p_args.device)
        elif p_args.fs == 'prod':
            self.fs_layer = FSLayerProduct().to(p_args.device)
        elif p_args.fs == "l2":
            self.fs_layer = FSLayerL2().to(p_args.device)
        self.rule_list = [RuleR(p_args.n_fea, p_args.n_class).to(p_args.device) for _ in range(p_args.n_rule)]
        for i, rules_item in enumerate(self.rule_list):
            self.add_module(f"rule_{i}", rules_item)

    def update_model(self, w_update):
        # # select local parameters from the global parameter
        # w_local = OrderedDict({})
        # for key, val in w_global.items():
        #     if "fs_layers" in key:
        #         w_local[key] = w_global[key]
        # for rule_idx in self.local_rule_idxs:
        #     for key, val in w_global.items():
        #         if f"local_rule_{rule_idx}" in key:
        #             w_local[key] = w_global[key]
        self.load_state_dict(w_update)

    def update_rules_idx_list(self, rules_idx_list):
        self.rules_idx_list = rules_idx_list

    def forward(self, data: torch.Tensor):
        n_batch = data.shape[0]
        # activate prototypes
        # produce antecedent layer
        fuzzy_set = torch.cat([rules_item.antecedent_layer(data).unsqueeze(0) for rules_item
                               in self.rule_list], dim=0)

        fire_strength_ini = torch.cat([self.fs_layer(data_diff_item) for data_diff_item in fuzzy_set], dim=1)
        fire_strength = F.softmax(fire_strength_ini[:, self.rules_idx_list], dim=1)

        # produce consequent layer
        data_processed = torch.cat([F.relu(rules_item.consequent_layer(data)).unsqueeze(0) for rules_item
                                    in self.rule_list], dim=0)
        data_processed = data_processed.view(self.args.n_rule, n_batch, self.args.n_class)

        outputs = torch.cat([(data_processed_item * fire_strength_item.unsqueeze(1)).unsqueeze(0)
                             for data_processed_item, fire_strength_item in
                             zip(data_processed[self.rules_idx_list], fire_strength.t())],
                            dim=0).sum(0)

        return outputs, fire_strength
