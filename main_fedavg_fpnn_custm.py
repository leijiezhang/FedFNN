import argparse
import os
from utils.logger import Logger
from data_process.dataset import FedDatasetCV, get_dataset_mat
# import sys
import numpy as np
import torch
import wandb
import scipy.io as io
# sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./")))
from models.fpnn import *
from models.fedfpnn_fedavg_api import FedAvgAPI


def add_args(p_parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    p_parser.add_argument('--model', type=str, default='fed_fpnn_csm', metavar='N',
                          help='neural network used in training')

    p_parser.add_argument('--dataset', type=str, default='gsad', metavar='N',
                          help='dataset used for training')

    p_parser.add_argument('--fs', type=str, default='l2', metavar='N',
                          help='firing strength layer')

    p_parser.add_argument('--partition_method', type=str, default='hetero', metavar='N',
                          help='how to partition the dataset on local workers')

    p_parser.add_argument('--partition_alpha', type=float, default=0.1, metavar='PA',
                          help='partition alpha (default: 0.5)')

    p_parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                          help='input batch size for training (default: 64)')

    p_parser.add_argument('--optimizer', type=str, default='adam',
                          help='SGD with momentum; adam')

    p_parser.add_argument('--criterion', type=str, default='bce',
                          help='the loss function')

    p_parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                          help='learning rate (default: 0.001)')

    p_parser.add_argument('--wd', help='weight decay parameter;', type=float, default=0.001)

    p_parser.add_argument('--n_client', type=int, default=5, metavar='NN',
                          help='number of workers in a distributed cluster')

    p_parser.add_argument('--n_client_per_round', type=int, default=5, metavar='NN',
                          help='number of workers')

    p_parser.add_argument('--comm_round', type=int, default=100,
                          help='how many round of communications we shoud use')

    p_parser.add_argument('--milestone', type=int, default=10,
                          help='the tag that local rules suit their own rules after certain round of communications')

    p_parser.add_argument('--epochs', type=int, default=15, metavar='EP',
                          help='how many epochs will be trained locally')

    p_parser.add_argument('--frequency_of_the_test', type=int, default=1,
                          help='the frequency of the algorithms')

    p_parser.add_argument('--gpu', type=int, default=0,
                          help='gpu')

    p_parser.add_argument('--n_rule', type=int, default=15,
                          help='rule number')
    p_parser.add_argument('--n_rule_min', type=int, default=5,
                          help='rule number')
    p_parser.add_argument('--n_rule_max', type=int, default=8,
                          help='rule number')

    p_parser.add_argument('--n_kernel', type=int, default=5,
                          help='Cov kernel number')

    p_parser.add_argument('--hidden_dim', type=int, default=512,
                          help='the output dim of the EEG encoder')

    p_parser.add_argument('--dropout', type=float, default=0.25, metavar='DR',
                          help='dropout rate (default: 0.025)')
    p_parser.add_argument(
        "--nl",
        type=float,
        default=0.0,
        help="noise level on dataset corruption",
    )
    p_parser.add_argument('--alpha', default=0., type=float,
                          help='mixup interpolation coefficient (default: 1)')

    p_parser.add_argument('--n_kfolds', type=int, default=5,
                          help='The number of k_fold cross validation')

    p_parser.add_argument('--b_debug', type=int, default=1,
                          help='set 1 to change to debug mode')

    p_parser.add_argument('--b_norm_dataset', type=int, default=1,
                          help='set 1 to change to debug mode')

    p_args = p_parser.parse_args()
    return p_args


def create_model(p_args):
    p_args.logger.info("======create_model.======")
    p_model: torch.nn.Module = None
    if p_args.model == "fpnn":
        local_rule_idxs = np.arange(p_args.n_rule)
        p_model: torch.nn.Module = FPNN(p_args, local_rule_idxs)
    elif p_args.model == "fed_fpnn_csm":
        # client_idx_list = np.arange(p_args.n_client)
        # initiate the global rule list
        # golobal_rule_list = [RuleR(p_args.n_fea, p_args.n_class, client_idx_list) for _ in range(p_args.n_rule)]
        local_rule_idxs = np.arange(p_args.n_rule)
        # global_fs_layer = FSLayer(p_args.n_fea, p_args.dropout)
        # p_model: torch.nn.Module = FedFPNNR(p_args.n_class, global_fs_layer, local_rule_idxs, golobal_rule_list)
        p_model: torch.nn.Module = FPNN(p_args, local_rule_idxs)

    return p_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameter parse of this project")
    args = add_args(parser)
    args.logger = Logger(True, args.dataset, args.model)
    args.device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")

    # Dataset configuration
    args.logger.info(f"========================={args.model}========================")
    args.logger.info(f"dataset : {args.dataset}")
    args.logger.info(f"device : {args.device}")
    args.logger.info(f"batch size : {args.batch_size}")
    args.logger.info(f"epoch number : {args.epochs}")
    args.logger.info(f"rule number : {args.n_rule}")

    global_train_acc_tsr = torch.zeros(args.comm_round, args.n_kfolds).to(args.device)
    global_train_loss_tsr = torch.zeros(args.comm_round, args.n_kfolds).to(args.device)
    global_test_loss_tsr = torch.zeros(args.comm_round, args.n_kfolds).to(args.device)
    global_test_acc_tsr = torch.zeros(args.comm_round, args.n_kfolds).to(args.device)

    global_train_rule_count = torch.zeros(args.comm_round, args.n_rule, args.n_kfolds).to(args.device)
    global_train_rule_contr = torch.zeros(args.comm_round, args.n_rule, args.n_kfolds).to(args.device)

    local_train_rule_count = torch.zeros(args.comm_round, args.n_client, args.n_rule, args.n_kfolds).to(args.device)
    local_train_rule_contr = torch.zeros(args.comm_round, args.n_client, args.n_rule, args.n_kfolds).to(args.device)

    local_train_acc_tsr = torch.zeros(args.comm_round, args.n_client, args.n_kfolds).to(args.device)
    local_train_loss_tsr = torch.zeros(args.comm_round, args.n_client, args.n_kfolds).to(args.device)
    local_test_loss_tsr = torch.zeros(args.comm_round, args.n_client, args.n_kfolds).to(args.device)
    local_test_acc_tsr = torch.zeros(args.comm_round, args.n_client, args.n_kfolds).to(args.device)

    n_para = 0

    for cv_idx in range(args.n_kfolds):
    # for cv_idx in range(1, 5):
        args.cv = cv_idx
        args.logger.war(f"=====k_fold: {cv_idx + 1}=======")

        # load data
        dir_dataset = f"./data/{args.dataset}/{args.dataset}.mat"
        dataset: FedDatasetCV = get_dataset_mat(dir_dataset, args)
        dataset.set_current_folds(cv_idx)

        # save category number
        args.n_class = dataset.n_class
        args.n_fea = dataset.n_fea

        args.tag = f"{args.dataset}_{args.model}_r{args.n_rule}" \
                   f"c{args.n_client}p{args.n_client_per_round}" \
                   f"_{args.partition_method}{args.partition_alpha}" \
                   f"_nl{args.nl}_{args.criterion}_lr{args.lr}_e{args.epochs}cr{args.comm_round}"

        if not args.b_debug:
            wandb.init(
                project=f"FederatedFPNN-{args.dataset}",
                name=f"{args.tag}_cv{cv_idx+1}",
                config=args
            )

        # create model.
        global_model = create_model(args)
        # args.logger.info(global_model)
        n_para = sum(param.numel() for param in global_model.parameters())
        args.logger.war(f"parameter amount : {n_para}")

        # federated method
        fedavgAPI = FedAvgAPI(dataset, global_model, args)
        metrics_list = fedavgAPI.train()

        for commu_idx in range(args.comm_round):
            metrics = metrics_list[commu_idx]
            global_train_acc_tsr[commu_idx, cv_idx] = metrics['training_acc']
            global_train_loss_tsr[commu_idx, cv_idx] = metrics['training_loss']
            global_test_loss_tsr[commu_idx, cv_idx] = metrics['test_loss']
            global_test_acc_tsr[commu_idx, cv_idx] = metrics['test_acc']
            for rule_idx in torch.arange(args.n_rule):
                global_train_rule_count[commu_idx, rule_idx, cv_idx] = metrics[f"rule{rule_idx + 1}_count"]
                global_train_rule_contr[commu_idx, rule_idx, cv_idx] = metrics[f"rule{rule_idx + 1}_contr"]
                for client_idx in range(args.n_client):
                    local_train_rule_count[commu_idx, client_idx, rule_idx, cv_idx] = \
                        metrics[f"client{client_idx + 1}_rule{rule_idx + 1}_count"]
                    local_train_rule_contr[commu_idx, client_idx, rule_idx, cv_idx] = \
                        metrics[f"client{client_idx + 1}_rule{rule_idx + 1}_contr"]
            for client_idx in range(args.n_client):
                local_train_acc_tsr[commu_idx, client_idx, cv_idx] = metrics[f"client{client_idx + 1}_train_acc"]
                local_train_loss_tsr[commu_idx, client_idx, cv_idx] = metrics[f"client{client_idx + 1}_train_loss"]
                local_test_acc_tsr[commu_idx, client_idx, cv_idx] = metrics[f"client{client_idx + 1}_test_acc"]
                local_test_loss_tsr[commu_idx, client_idx, cv_idx] = metrics[f"client{client_idx + 1}_test_loss"]
        if not args.b_debug:
            wandb.finish()

    save_dict = dict()
    save_dict["global_test_acc_tsr"] = global_test_acc_tsr.cpu().numpy()
    save_dict["global_test_loss_tsr"] = global_test_loss_tsr.cpu().numpy()
    save_dict["global_train_loss_tsr"] = global_train_loss_tsr.cpu().numpy()
    save_dict["global_train_acc_tsr"] = global_train_acc_tsr.cpu().numpy()
    save_dict["global_train_rule_count"] = global_train_rule_count.cpu().numpy()
    save_dict["global_train_rule_contr"] = global_train_rule_contr.cpu().numpy()

    save_dict["local_test_acc_tsr"] = local_test_acc_tsr.cpu().numpy()
    save_dict["local_test_loss_tsr"] = local_test_loss_tsr.cpu().numpy()
    save_dict["local_train_loss_tsr"] = local_train_loss_tsr.cpu().numpy()
    save_dict["local_train_acc_tsr"] = local_train_acc_tsr.cpu().numpy()
    save_dict["local_train_rule_count"] = local_train_rule_count.cpu().numpy()
    save_dict["local_train_rule_contr"] = local_train_rule_contr.cpu().numpy()

    save_dict["n_para"] = n_para

    save_file_name = f"{args.tag}.mat"

    data_save_dir = f"./results/{args.dataset}"

    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)

    data_save_file = f"{data_save_dir}/{save_file_name}"

    io.savemat(data_save_file, save_dict)
