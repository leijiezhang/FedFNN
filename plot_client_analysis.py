import scipy.io as sio
import torch
import os
import scipy.io as io
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns


# Dataset configuration

colors = ["#9e97cb", "#4586ac", "#cb5a48", "#3498db", "#95a5a6", "#e74c3c"]
palette = sns.color_palette(colors[1:2])
# colors = ["#2ecc71", "#9b59b6", "#DDA0DD", "#3498db", "#87CEFA", "#95a5a6","#e74c3c", "#95a5a6", "#9b59b6", "#B8860B"]
colors1 = ["#2ecc71", "#9b59b6", "#3498db", "#95a5a6","#e74c3c", "#95a5a6", "#9b59b6", "#B8860B"]

alg = ["FedDNN+CCVR+MOON (*)", "FedDNN (*)", "FedFNN (*)"]

client_list = ['1', '2', '3', '4', '5']
# get dataset sample number
load_path = f"./results/client_analysis/gsad_fed_fpnn_csm_n_smpl_r15c5p5_hetero0.1_nl0.0_bce_lr0.001_e15cr100.mat"
load_data = sio.loadmat(load_path)
n_sampl_cat_tbl = torch.tensor(load_data['n_sampl_cat_tbl'])
n_sampl_cat_tbl = n_sampl_cat_tbl.reshape(30).numpy()
category_list = ['category1', 'category2', 'category3', 'category4', 'category5', 'category6']*5
client_list_n_smpl = ['Client1']*6+['Client2']*6+['Client3']*6+['Client4']*6+['Client5']*6

n_smpl_data = []
for i in range(n_sampl_cat_tbl.shape[0]):
    n_smpl_data.append([client_list_n_smpl[i], n_sampl_cat_tbl[i], category_list[i]])
n_smpl_data_pd = DataFrame(n_smpl_data, columns=["client", 'Sample Number', 'Category'])

# get local performance
fed_dnn_ccvr_hetero_train = torch.zeros(5, len(client_list))
fed_dnn_hetero_train = torch.zeros(5, len(client_list))
fed_fnn_train = torch.zeros(5, len(client_list))
fed_dnn_ccvr_hetero_test = torch.zeros(5, len(client_list))
fed_dnn_hetero_test = torch.zeros(5, len(client_list))
fed_fnn_test = torch.zeros(5, len(client_list))

load_path = f"./results/client_analysis/gsad_client_analysis.mat"
load_data = sio.loadmat(load_path)
fed_dnn_ccvr_hetero_train[:, :] = torch.tensor(load_data['fed_dnn_ccvr_hetero_train'])
fed_dnn_hetero_train[:, :] = torch.tensor(load_data['fed_dnn_hetero_train'])
fed_fnn_train[:, :] = torch.tensor(load_data['fed_fnn_train'])
fed_dnn_ccvr_hetero_test[:, :] = torch.tensor(load_data['fed_dnn_ccvr_hetero_test'])
fed_dnn_hetero_test[:, :] = torch.tensor(load_data['fed_dnn_hetero_test'])
fed_fnn_test[:, :] = torch.tensor(load_data['fed_fnn_test'])
    

train_data = torch.cat([fed_dnn_hetero_train, fed_dnn_ccvr_hetero_train], 1)
train_data = torch.cat([train_data, fed_fnn_train], 1)
train_data = train_data.view(-1).numpy()

test_data = torch.cat([fed_dnn_hetero_test, fed_dnn_ccvr_hetero_test], 1)
test_data = torch.cat([test_data, fed_fnn_test], 1)
test_data = test_data.view(-1).numpy()

alg_list = (["FedDNN (*)"]*5+['FedDNN+CCVR+MOON (*)']*5+["FedFNN (*)"]*5)*5
client_list = ['Client1', 'Client2', 'Client3', 'Client4', 'Client5']*25

client_data = []
for i in range(train_data.shape[0]):
    client_data.append([client_list[i], train_data[i], test_data[i], alg_list[i]])
client_data_pd = DataFrame(client_data, columns=["client", 'Train Accuracy', 'Test Accuracy', "Algorithm"])


# ================plot hetegeneouety trend figure====================
plt.rcParams['figure.figsize']=[6, 20]
plt.subplots_adjust(wspace=0.3, hspace=0.3)
ax1 = plt.subplot(211)
# sns.barplot(x="client", y='Train Accuracy', hue="Algorithm", data=client_data_pd)
# ax1.set_ylabel('Train Accuracy', fontsize=10)
# ax1.set_xlabel('', fontsize=10)
# ax1.set_ylabel('Train Accuracy', fontsize=10)
# ax1.set_xlabel('', fontsize=10)
# ax1.legend_.remove()
sns.barplot(x="client", y='Sample Number', hue="Category", data=n_smpl_data_pd)
ax1.set_ylabel('Sample Number', fontsize=13)
ax1.set_xlabel('', fontsize=13)

ax2 = plt.subplot(212)
sns.barplot(x="client", y='Test Accuracy', hue="Algorithm", data=client_data_pd)
ax2.set_ylabel('Test Accuracy', fontsize=13)
ax2.set_xlabel('', fontsize=13)
ax2.legend_.remove()

ax1.legend(fontsize=12, loc='upper center', bbox_to_anchor=(0.84, 1.02), ncol=1)

plt.savefig(f"./results/client_analysis_v.pdf", bbox_inches='tight')
print("lslsl")