import scipy.io as sio
import torch
import os
import scipy.io as io
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns


# Dataset configuration

colors = ["#9e97cb", "#4586ac", "#cb5a48", "#3498db", "#95a5a6", "#e74c3c"]
# colors = ["#2ecc71", "#9b59b6", "#DDA0DD", "#3498db", "#87CEFA", "#95a5a6","#e74c3c", "#95a5a6", "#9b59b6", "#B8860B"]
colors1 = ["#2ecc71", "#9b59b6", "#3498db", "#95a5a6","#e74c3c", "#95a5a6", "#9b59b6", "#B8860B"]

alg = ["MOON (*)", "FEDAVG (*)", "FedFNN (*)"]

alpha_list = ['0.1', '0.5', '5', '20', '50', '100']

fed_dnn_ccvr_hetero = torch.zeros(5, len(alpha_list))
fed_dnn_hetero = torch.zeros(5, len(alpha_list))
fed_fnn = torch.zeros(5, len(alpha_list))

for j in torch.arange(len(alpha_list)):
    load_path = f"./results/hetero_trend/gsad_hetero_trend.mat"
    load_data = sio.loadmat(load_path)
    fed_dnn_ccvr_hetero[:, :] = torch.tensor(load_data['fed_dnn_ccvr_hetero'])
    fed_dnn_hetero[:, :] = torch.tensor(load_data['fed_dnn_hetero'])
    fed_fnn[:, :] = torch.tensor(load_data['fed_fnn'])

fed_dnn_ccvr_hetero = fed_dnn_ccvr_hetero.view(-1).numpy()
fed_dnn_hetero = fed_dnn_hetero.view(-1).numpy()
fed_fnn = fed_fnn.view(-1).numpy()

#plot noise  trend
# alpha_item = torch.tensor([0.1, 0.5, 5, 20, 50, 100])
alpha_item = torch.tensor([1, 2, 3, 4, 5, 6])
alpha = alpha_item.repeat([5]).numpy()

gsad_trend_data = []
for i in range(alpha.shape[0]):
    gsad_trend_data.append([alpha[i], fed_dnn_ccvr_hetero[i], fed_dnn_hetero[i], fed_fnn[i]])
gsad_trend = DataFrame(gsad_trend_data, columns=["alpha", 'MOON (*)', "FEDAVG (*)", "FedFNN (*)"])


# ================plot hetegeneouety trend figure====================
plt.rcParams['figure.figsize']=[16, 8]
plt.subplots_adjust(wspace=0.3, hspace=0.3)
ax1 = plt.subplot(111)
for i in range(len(alg)):
    a = alg[i]
    c = colors[i]
    sns.lineplot(x="alpha", y=a, data=gsad_trend, color=c, ci='sd', label=a)
tick_label = ['', '0.1', '0.5', '5', '20', '50', '100']
ax1.set_xticklabels(tick_label)
ax1.set_ylabel('Test Accuracy', fontsize=28)
ax1.set_xlabel('Alpha', fontsize=28)
plt.yticks(size=26)
plt.xticks(size=26)
ax1.legend(fontsize=35, loc='upper center', bbox_to_anchor=(0.48, 1.19), ncol=3)
plt.savefig(f"./results/hetero_trend.pdf", bbox_inches='tight')
print("lslsl")