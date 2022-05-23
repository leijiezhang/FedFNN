import argparse
import os
import numpy as np
import torch
import wandb
from utils.logger import Logger
import scipy.io as io
from data_process.dataset import FedDatasetCV, get_dataset_mat
from torch.nn.parallel import DistributedDataParallel
# sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./")))
from models.fpnn_centr_trainer import CentralizedTrainer
from models.fpnn import FPNN


def add_args(p_parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    p_parser.add_argument('--model', type=str, default='cen_fpnn', metavar='N',
                          help='neural network used in training')

    p_parser.add_argument('--data_parallel', type=int, default=0,
                          help='if distributed training')

    p_parser.add_argument('--dataset', type=str, default='gsad', metavar='N',
                          help='dataset used for training')

    p_parser.add_argument('--fs', type=str, default='sum', metavar='N',
                          help='firing strength layer')

    p_parser.add_argument('--time_samples', type=int, default=750,
                          help='time sample number')

    p_parser.add_argument('--client_num_in_total', type=int, default=9, metavar='NN',
                          help='number of workers in a distributed cluster')

    p_parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                          help='input batch size for training (default: 64)')

    p_parser.add_argument('--optimizer', type=str, default='adam',
                          help='SGD with momentum; adam')

    p_parser.add_argument('--criterion', type=str, default='bce',
                          help='the loss function')

    p_parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                          help='learning rate (default: 0.001)')

    p_parser.add_argument('--drop_rate', type=float, default=0.25, metavar='DR',
                          help='dropout rate (default: 0.025)')

    p_parser.add_argument('--wd', help='weight decay parameter;', type=float, default=0.001)

    p_parser.add_argument('--epochs', type=int, default=50, metavar='EP',
                          help='how many epochs will be trained locally')

    p_parser.add_argument('--frequency_of_train_acc_report', type=int, default=1,
                          help='the frequency of training accuracy report')

    p_parser.add_argument('--frequency_of_test_acc_report', type=int, default=1,
                          help='the frequency of test accuracy report')

    p_parser.add_argument('--gpu_server_num', type=int, default=1,
                          help='gpu_server_num')

    p_parser.add_argument('--gpu', type=int, default=0,
                          help='gpu')

    p_parser.add_argument('--n_rule', type=int, default=5,
                          help='rule number')

    p_parser.add_argument('--hidden_dim', type=int, default=16,
                          help='the output dim of the EEG encoder')

    p_parser.add_argument('--partition_method', type=str, default='homo', metavar='N',
                          help='how to partition the dataset on local workers')

    p_parser.add_argument('--partition_alpha', type=float, default=0.5, metavar='PA',
                          help='partition alpha (default: 0.5)')
    p_parser.add_argument(
        "--nl",
        type=float,
        default=0.0,
        help="noise level on dataset corruption",
    )

    p_parser.add_argument('--n_client', type=int, default=5, metavar='NN',
                          help='number of workers in a distributed cluster')

    p_parser.add_argument('--n_kfolds', type=int, default=4,
                          help='The number of k_fold cross validation')

    p_parser.add_argument('--b_debug', type=int, default=1,
                          help='set 1 to change to debug mode')

    p_parser.add_argument('--b_norm_dataset', type=int, default=1,
                          help='set 1 to change to debug mode')

    p_parser.add_argument('--dropout', type=float, default=0.0, metavar='DR',
                          help='dropout rate (default: 0.025)')
    p_args = p_parser.parse_args()
    return p_args


def create_model(p_args):
    p_args.logger.info("======create_model.======")
    p_model: torch.nn.Module = None
    if p_args.model == "cen_fpnn":
        local_rule_idxs = np.arange(p_args.n_rule)
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

    global_train_acc_tsr = torch.zeros(args.epochs, args.n_kfolds).to(args.device)
    global_train_loss_tsr = torch.zeros(args.epochs, args.n_kfolds).to(args.device)
    global_test_loss_tsr = torch.zeros(args.epochs, args.n_kfolds).to(args.device)
    global_test_acc_tsr = torch.zeros(args.epochs, args.n_kfolds).to(args.device)

    global_train_rule_count = torch.zeros(args.epochs, args.n_rule, args.n_kfolds).to(args.device)
    global_train_rule_contr = torch.zeros(args.epochs, args.n_rule, args.n_kfolds).to(args.device)

    for cv_idx in range(args.n_kfolds):
    # for cv_idx in range(1):
        args.cv = cv_idx
        args.logger.war(f"=====k_fold: {cv_idx + 1}=======")

        # load data + str(args.dataset)
        dataset: FedDatasetCV = get_dataset_mat(args.dataset, args)
        dataset.set_current_folds(cv_idx)

        # save category number
        args.n_class = dataset.n_class
        args.n_fea = dataset.n_fea
        if args.partition_method == 'homo':
            args.tag = f"{args.model}_{args.n_rule}_{args.partition_method}_{args.nl}_{args.criterion}_{args.lr}"
        else:
            args.tag = f"{args.model}_{args.n_rule}_{args.n_client}_{args.n_client_per_round}_{args.partition_method}" \
                    f"_{args.partition_alpha}" \
                    f"_{args.nl}_{args.criterion}_{args.lr}"

        if not args.b_debug:
            wandb.init(
                project=f"FederatedFPNN-{args.dataset}",
                name=str(args.model) +
                     "-r" + str(args.n_rule) +
                     "-c" + str(args.n_client) +
                     "-nl" + str(args.nl) +
                     "-" + args.criterion +
                     "-lr" + str(args.lr) +
                     "-cv" + str(cv_idx + 1),
                config=args
            )

        # create model.
        model = create_model(args)
        # args.logger.info(model)
        # start "federated averaging (FedAvg)"
        single_trainer = CentralizedTrainer(dataset, model, model.rules_idx_list, args)
        metrics_list = single_trainer.train()

        for epoch_idx in range(args.epochs):
            metrics = metrics_list[epoch_idx]
            global_train_acc_tsr[epoch_idx, cv_idx] = metrics['training_acc']
            global_train_loss_tsr[epoch_idx, cv_idx] = metrics['training_loss']
            global_test_loss_tsr[epoch_idx, cv_idx] = metrics['test_loss']
            global_test_acc_tsr[epoch_idx, cv_idx] = metrics['test_acc']
            for rule_idx in torch.arange(args.n_rule):
                global_train_rule_count[epoch_idx, rule_idx, cv_idx] = metrics[f"rule{rule_idx + 1}_count"]
                global_train_rule_contr[epoch_idx, rule_idx, cv_idx] = metrics[f"rule{rule_idx + 1}_contr"]

        if not args.b_debug:
            wandb.finish()

    save_dict = dict()
    save_dict["global_test_acc_tsr"] = global_test_acc_tsr.cpu().numpy()
    save_dict["global_test_loss_tsr"] = global_test_loss_tsr.cpu().numpy()
    save_dict["global_train_loss_tsr"] = global_train_loss_tsr.cpu().numpy()
    save_dict["global_train_acc_tsr"] = global_train_acc_tsr.cpu().numpy()
    save_dict["global_train_rule_count"] = global_train_rule_count.cpu().numpy()
    save_dict["global_train_rule_contr"] = global_train_rule_contr.cpu().numpy()

    save_file_name = "cen" + str(args.dataset) + "-r" + str(args.n_rule) + "-c" + str(args.n_client) \
                     + "-nl" + str(args.nl) + "-" + args.criterion \
                     + "-lr" + str(args.lr) + ".mat"

    data_save_dir = f"./results"

    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)

    data_save_file = f"{data_save_dir}/{save_file_name}"

    io.savemat(data_save_file, save_dict)
