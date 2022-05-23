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
from models.dnn_model import *
from models.dnn_fedavg_api import FedAvgAPI
from models.dnn_fed_trainer import MyModelTrainer as MyModelTrainerDNN


def add_args(p_parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    p_parser.add_argument('--model', type=str, default='mlp', metavar='N',
                          help='neural network used in training')

    p_parser.add_argument('--dataset', type=str, default='wifi', metavar='N',
                          help='dataset used for training')

    p_parser.add_argument('--partition_method', type=str, default='hetero', metavar='N',
                          help='how to partition the dataset on local workers')

    p_parser.add_argument('--partition_alpha', type=float, default=0.5, metavar='PA',
                          help='partition alpha (default: 0.5)')

    p_parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                          help='input batch size for training (default: 64)')

    p_parser.add_argument('--optimizer', type=str, default='adam',
                          help='SGD with momentum; adam')

    p_parser.add_argument('--criterion', type=str, default='ce',
                          help='the loss function')

    p_parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                          help='learning rate (default: 0.001)')

    p_parser.add_argument('--wd', help='weight decay parameter;', type=float, default=0.001)

    p_parser.add_argument('--epochs', type=int, default=5, metavar='EP',
                          help='how many epochs will be trained locally')

    p_parser.add_argument('--n_client', type=int, default=5, metavar='NN',
                          help='number of workers in a distributed cluster')

    p_parser.add_argument('--n_client_per_round', type=int, default=5, metavar='NN',
                          help='number of workers')

    p_parser.add_argument('--comm_round', type=int, default=100,
                          help='how many round of communications we shoud use')

    p_parser.add_argument('--gpu', type=int, default=1,
                          help='gpu')

    p_parser.add_argument('--output_dim', type=int, default=256,
                          help='the output dim of the EEG encoder')

    p_parser.add_argument('--dropout', type=float, default=0.1, metavar='DR',
                          help='dropout rate (default: 0.025)')

    p_parser.add_argument('--gnia_sig', type=float, default=0.05, metavar='DR',
                          help='gnia_sig (default: 0.025)')

    p_parser.add_argument(
        "--nl",
        type=float,
        default=0.2,
        help="noise level on dataset corruption",
    )

    p_parser.add_argument('--n_kfolds', type=int, default=5,
                          help='The number of k_fold cross validation')

    p_parser.add_argument('--b_debug', type=int, default=1,
                          help='set 1 to change to debug mode')

    p_parser.add_argument('--b_norm_dataset', type=int, default=1,
                          help='set 1 to change to debug mode')

    p_args = p_parser.parse_args()
    return p_args


def create_model_structure(p_args):
    model_strc = [[p_args.n_fea, 2*p_args.n_fea, 4*p_args.n_fea, 8*p_args.n_fea],
                  [p_args.n_fea, 4*p_args.n_fea, 2*p_args.n_fea],
                  [p_args.n_fea, 8*p_args.n_fea, 4*p_args.n_fea, 2*p_args.n_fea],
                  [p_args.n_fea, 2*p_args.n_fea, 4*p_args.n_fea, 8*p_args.n_fea, 4*p_args.n_fea],
                  [p_args.n_fea, 256, 512, 256],
                  [p_args.n_fea, 256, 512, 256, 128],
                  [p_args.n_fea, 512, 512, 256],
                  [p_args.n_fea, 256, 512, 1024],
                  [p_args.n_fea, 256, 512, 1024, 512],
                  [p_args.n_fea, 64, 128, 64],
                  [p_args.n_fea, 128, 64, 64],
                  [p_args.n_fea, 64, 64, 32],
                  [p_args.n_fea, 64, 64, 32, 32]
                  ]
    return model_strc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameter parse of this project")
    args = add_args(parser)
    # for model_idx in range(13):
    for model_idx in [8]:
        args.logger = Logger(True, args.dataset, args.model)
        args.device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")

        global_train_acc_tsr = torch.zeros(args.comm_round, args.n_kfolds).to(args.device)
        global_train_loss_tsr = torch.zeros(args.comm_round, args.n_kfolds).to(args.device)
        global_test_loss_tsr = torch.zeros(args.comm_round, args.n_kfolds).to(args.device)
        global_test_acc_tsr = torch.zeros(args.comm_round, args.n_kfolds).to(args.device)

        local_train_acc_tsr = torch.zeros(args.comm_round, args.n_client, args.n_kfolds).to(args.device)
        local_train_loss_tsr = torch.zeros(args.comm_round, args.n_client, args.n_kfolds).to(args.device)
        local_test_loss_tsr = torch.zeros(args.comm_round, args.n_client, args.n_kfolds).to(args.device)
        local_test_acc_tsr = torch.zeros(args.comm_round, args.n_client, args.n_kfolds).to(args.device)

        n_para = 0

        for cv_idx in range(args.n_kfolds):
            # Dataset configuration
            args.logger.info(f"========================={args.model}_{model_idx+1}_fedavg_({cv_idx+1}/{args.n_kfolds})"
                             f"========================")
            args.logger.info(f"dataset : {args.dataset}")
            args.logger.info(f"device : {args.device}")
            args.logger.info(f"batch size : {args.batch_size}")
            args.logger.info(f"epoch number : {args.epochs}")
            args.logger.info(f"comm round : {args.comm_round}")

            # for cv_idx in range(1):
            args.cv = cv_idx
            # load data
            dir_dataset = f"./data/{args.dataset}/{args.dataset}.mat"
            dataset: FedDatasetCV = get_dataset_mat(dir_dataset, args)
            dataset.set_current_folds(cv_idx)

            # save category number
            args.n_class = dataset.n_class
            args.n_fea = dataset.n_fea
            args.tag = f"{args.dataset}_{args.model}{model_idx+1}_fedavg_{args.output_dim}_" \
                       f"c{args.n_client}p{args.n_client_per_round}" \
                       f"_{args.partition_method}{args.partition_alpha}" \
                       f"_s{args.gnia_sig}_d{args.dropout}" \
                       f"_nl{args.nl}_{args.criterion}_lr{args.lr}_e{args.epochs}cr{args.comm_round}"

            if not args.b_debug:
                wandb.init(
                    project=f"DNN",
                    name=f"{args.tag}_cv{cv_idx+1}",
                    config=args
                )

            # create model.
            model_strc = create_model_structure(args)
            model = MyMLP(args.model, hidden_dims=model_strc[model_idx], out_dim=args.output_dim, n_classes=args.n_class,
                          dropout_p=args.dropout)
            n_para = sum(param.numel() for param in model.parameters())
            args.logger.war(f"parameter amount : {n_para}")
            model_trainer = MyModelTrainerDNN(model, args)

            # federated method
            fedavgAPI = FedAvgAPI(dataset, model_trainer, args)
            metrics_list = fedavgAPI.train()

            for commu_idx in range(args.comm_round):
                metrics = metrics_list[commu_idx]
                global_train_acc_tsr[commu_idx, cv_idx] = metrics['training_acc']
                global_train_loss_tsr[commu_idx, cv_idx] = metrics['training_loss']
                global_test_loss_tsr[commu_idx, cv_idx] = metrics['test_loss']
                global_test_acc_tsr[commu_idx, cv_idx] = metrics['test_acc']
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

        save_dict["local_test_acc_tsr"] = local_test_acc_tsr.cpu().numpy()
        save_dict["local_test_loss_tsr"] = local_test_loss_tsr.cpu().numpy()
        save_dict["local_train_loss_tsr"] = local_train_loss_tsr.cpu().numpy()
        save_dict["local_train_acc_tsr"] = local_train_acc_tsr.cpu().numpy()

        save_dict["n_para"] = n_para

        save_file_name = f"{args.tag}.mat"

        data_save_dir = f"./results/{args.dataset}"

        if not os.path.exists(data_save_dir):
            os.makedirs(data_save_dir)

        data_save_file = f"{data_save_dir}/{save_file_name}"

        io.savemat(data_save_file, save_dict)
