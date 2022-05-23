import scipy.io as sio
import torch
import os
import scipy.io as io
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns


# Dataset configuration
dataset_list = ['sdd', 'gsad', 'meterd', 'magic',
                'shuttle', 'robot', 'wifi']
rules_list = [15, 15, 15, 15, 15, 15, 15]
epoch_list = [100, 100, 100, 100, 100, 100, 100]

colors = ["#DDA0DD", "#9b59b6", "#2ecc71", "#3498db", "#95a5a6", "#e74c3c"]
# colors = ["#2ecc71", "#9b59b6", "#DDA0DD", "#3498db", "#87CEFA", "#95a5a6","#e74c3c", "#95a5a6", "#9b59b6", "#B8860B"]
colors1 = ["#2ecc71", "#9b59b6", "#3498db", "#95a5a6","#e74c3c", "#95a5a6", "#9b59b6", "#B8860B"]

alg = ["DFNN (+)", 'DFNN (*)', "FedDNN+CCVR+MOON (*)", "FedDNN (*)", "FedDNN (+)", "FedFNN (*)"]

uncertainty_level = ['0.0', '0.1', '0.2', '0.3']

n_dataset = len(dataset_list)
dfnn_homo = torch.zeros(len(dataset_list), 5, 4)
dfnn_hetero = torch.zeros(len(dataset_list), 5, 4)
fed_dnn_ccvr_hetero = torch.zeros(len(dataset_list), 5, 4)
fed_dnn_hetero = torch.zeros(len(dataset_list), 5, 4)
fed_dnn_homo = torch.zeros(len(dataset_list), 5, 4)
fed_fnn = torch.zeros(len(dataset_list), 5, 4)

fed_fnn_trend_dict = dict()

for i in torch.arange(n_dataset):
    fed_fnn_trend_tsr = torch.zeros(4, 5, epoch_list[i])
    for j in torch.arange(len(uncertainty_level)):
        load_data_dir = f"{dataset_list[i]}_fed_fpnn_csm_r15c5p5_hetero0.5_nl{uncertainty_level[j]}_ce_lr0.001_e15cr100.mat"
        load_path = f"./results/{dataset_list[i]}/{load_data_dir}"
        load_data = sio.loadmat(load_path)
        fed_fnn_test_acc = torch.tensor(load_data['global_test_acc_tsr'])
        fed_fnn_trend_tsr[j, :, :] = fed_fnn_test_acc.t()

    load_data_dir = f"{dataset_list[i]}_uncertainty_trend.mat"
    load_path = f"./results/uncertainty_trend/{load_data_dir}"
    load_data = sio.loadmat(load_path)
    dfnn_homo[i, :, :] = torch.tensor(load_data['dfnn_homo'])
    dfnn_hetero[i, :, :] = torch.tensor(load_data['dfnn_hetero'])
    fed_dnn_ccvr_hetero[i, :, :] = torch.tensor(load_data['fed_dnn_ccvr_hetero'])
    fed_dnn_hetero[i, :, :] = torch.tensor(load_data['fed_dnn_hetero'])
    fed_dnn_homo[i, :, :] = torch.tensor(load_data['fed_dnn_homo'])
    fed_fnn[i, :, :] = torch.tensor(load_data['fed_fnn'])


    fed_fnn_trend_dict[dataset_list[i]] = fed_fnn_trend_tsr.view(4, -1)

dfnn_homo = dfnn_homo.view(n_dataset, -1).numpy()
dfnn_hetero = dfnn_hetero.view(n_dataset, -1).numpy()
fed_dnn_ccvr_hetero = fed_dnn_ccvr_hetero.view(n_dataset, -1).numpy()
fed_dnn_hetero = fed_dnn_hetero.view(n_dataset, -1).numpy()
fed_dnn_homo = fed_dnn_homo.view(n_dataset, -1).numpy()
fed_fnn = fed_fnn.view(n_dataset, -1).numpy()

#plot noise  trend
noise_item = torch.tensor([0, 10, 20, 30])
noise = noise_item.repeat([5]).numpy()

sdd_trend_data = []
for i in range(noise.shape[0]):
    sdd_trend_data.append([noise[i], dfnn_homo[0][i], dfnn_hetero[0][i], fed_dnn_ccvr_hetero[0][i],
                           fed_dnn_hetero[0][i], fed_dnn_homo[0][i], fed_fnn[0][i]])

sdd_trend = DataFrame(sdd_trend_data,
                      columns=["uncertainty_level", "DFNN (+)", "DFNN (*)", 'FedDNN+CCVR+MOON (*)', "FedDNN (*)", "FedDNN (+)",
                               "FedFNN (*)"])
sdd_fed_fnn_trend_data = []
sdd_fed_fnn_trend_tsr = fed_fnn_trend_dict['sdd']
epoch_item = torch.arange(int(sdd_fed_fnn_trend_tsr.shape[1]/5)) + 1
epoch_num = epoch_item.repeat([5]).numpy()
for i in range(epoch_num.shape[0]):
    sdd_fed_fnn_trend_data.append([epoch_num[i], sdd_fed_fnn_trend_tsr.numpy()[0][i], sdd_fed_fnn_trend_tsr.numpy()[1][i],
                               sdd_fed_fnn_trend_tsr.numpy()[2][i], sdd_fed_fnn_trend_tsr.numpy()[3][i]])
sdd_fed_fnn_trend = DataFrame(sdd_fed_fnn_trend_data, columns=["epoch", "00%", '10%', "20%", "30%"])

gsad_trend_data = []
for i in range(noise.shape[0]):
    gsad_trend_data.append([noise[i], dfnn_homo[1][i], dfnn_hetero[1][i], fed_dnn_ccvr_hetero[1][i],
                           fed_dnn_hetero[1][i], fed_dnn_homo[1][i], fed_fnn[1][i]])
gsad_trend = DataFrame(gsad_trend_data, columns=["uncertainty_level", "DFNN (+)", "DFNN (*)", 'FedDNN+CCVR+MOON (*)', "FedDNN (*)", "FedDNN (+)",
                               "FedFNN (*)"])
gsad_fed_fnn_trend_data = []
gsad_fed_fnn_trend_tsr = fed_fnn_trend_dict['gsad']
epoch_item = torch.arange(int(gsad_fed_fnn_trend_tsr.shape[1]/5)) + 1
epoch_num = epoch_item.repeat([5]).numpy()
for i in range(epoch_num.shape[0]):
    gsad_fed_fnn_trend_data.append([epoch_num[i], gsad_fed_fnn_trend_tsr.numpy()[0][i], gsad_fed_fnn_trend_tsr.numpy()[1][i],
                               gsad_fed_fnn_trend_tsr.numpy()[2][i], gsad_fed_fnn_trend_tsr.numpy()[3][i]])
gsad_fed_fnn_trend = DataFrame(gsad_fed_fnn_trend_data, columns=["epoch", "00%", '10%', "20%", "30%"])

meterd_trend_data = []
for i in range(noise.shape[0]):
    meterd_trend_data.append([noise[i], dfnn_homo[2][i], dfnn_hetero[2][i], fed_dnn_ccvr_hetero[2][i],
                           fed_dnn_hetero[2][i], fed_dnn_homo[2][i], fed_fnn[2][i]])
meterd_trend = DataFrame(meterd_trend_data, columns=["uncertainty_level", "DFNN (+)", "DFNN (*)", 'FedDNN+CCVR+MOON (*)', "FedDNN (*)", "FedDNN (+)",
                               "FedFNN (*)"])
meterd_fed_fnn_trend_data = []
meterd_fed_fnn_trend_tsr = fed_fnn_trend_dict['meterd']
epoch_item = torch.arange(int(meterd_fed_fnn_trend_tsr.shape[1]/5)) + 1
epoch_num = epoch_item.repeat([5]).numpy()
for i in range(epoch_num.shape[0]):
    meterd_fed_fnn_trend_data.append([epoch_num[i], meterd_fed_fnn_trend_tsr.numpy()[0][i], meterd_fed_fnn_trend_tsr.numpy()[1][i],
                               meterd_fed_fnn_trend_tsr.numpy()[2][i], meterd_fed_fnn_trend_tsr.numpy()[3][i]])
meterd_fed_fnn_trend = DataFrame(meterd_fed_fnn_trend_data, columns=["epoch", "00%", '10%', "20%", "30%"])

# wine_trend_data = []
# for i in range(noise.shape[0]):
#     wine_trend_data.append([noise[i], dfnn_homo[1][i], dfnn_hetero[1][i], fed_dnn_ccvr_hetero[1][i],
#                            fed_dnn_hetero[1][i], fed_dnn_homo[1][i], fed_fnn[1][i]])
# wine_trend = DataFrame(wine_trend_data, columns=["uncertainty_level", "DFNN (+)", "DFNN (*)", 'FedDNN+CCVR+MOON (*)', "FedDNN (*)", "FedDNN (+)",
#                                "FedFNN (*)"])
# wine_fed_fnn_trend_data = []
# wine_fed_fnn_trend_tsr = fed_fnn_trend_dict['wine']
# epoch_item = torch.arange(int(wine_fed_fnn_trend_tsr.shape[1]/5)) + 1
# epoch_num = epoch_item.repeat([5]).numpy()
# for i in range(epoch_num.shape[0]):
#     wine_fed_fnn_trend_data.append([epoch_num[i], wine_fed_fnn_trend_tsr.numpy()[0][i], wine_fed_fnn_trend_tsr.numpy()[1][i],
#                                wine_fed_fnn_trend_tsr.numpy()[2][i], wine_fed_fnn_trend_tsr.numpy()[3][i]])
# wine_fed_fnn_trend = DataFrame(wine_fed_fnn_trend_data, columns=["epoch", "00%", '10%', "20%", "30%"])

magic_trend_data = []
for i in range(noise.shape[0]):
    magic_trend_data.append([noise[i], dfnn_homo[3][i], dfnn_hetero[3][i], fed_dnn_ccvr_hetero[3][i],
                           fed_dnn_hetero[3][i], fed_dnn_homo[3][i], fed_fnn[3][i]])
magic_trend = DataFrame(magic_trend_data, columns=["uncertainty_level", "DFNN (+)", "DFNN (*)", 'FedDNN+CCVR+MOON (*)', "FedDNN (*)", "FedDNN (+)",
                               "FedFNN (*)"])
magic_fed_fnn_trend_data = []
magic_fed_fnn_trend_tsr = fed_fnn_trend_dict['magic']
epoch_item = torch.arange(int(magic_fed_fnn_trend_tsr.shape[1]/5)) + 1
epoch_num = epoch_item.repeat([5]).numpy()
for i in range(epoch_num.shape[0]):
    magic_fed_fnn_trend_data.append([epoch_num[i], magic_fed_fnn_trend_tsr.numpy()[0][i], magic_fed_fnn_trend_tsr.numpy()[1][i],
                               magic_fed_fnn_trend_tsr.numpy()[2][i], magic_fed_fnn_trend_tsr.numpy()[3][i]])
magic_fed_fnn_trend = DataFrame(magic_fed_fnn_trend_data, columns=["epoch", "00%", '10%', "20%", "30%"])

shuttle_trend_data = []
for i in range(noise.shape[0]):
    shuttle_trend_data.append([noise[i], dfnn_homo[4][i], dfnn_hetero[4][i], fed_dnn_ccvr_hetero[4][i],
                           fed_dnn_hetero[4][i], fed_dnn_homo[4][i], fed_fnn[4][i]])
shuttle_trend = DataFrame(shuttle_trend_data, columns=["uncertainty_level", "DFNN (+)", "DFNN (*)", 'FedDNN+CCVR+MOON (*)', "FedDNN (*)", "FedDNN (+)",
                               "FedFNN (*)"])
shuttle_fed_fnn_trend_data = []
shuttle_fed_fnn_trend_tsr = fed_fnn_trend_dict['shuttle']
epoch_item = torch.arange(int(shuttle_fed_fnn_trend_tsr.shape[1]/5)) + 1
epoch_num = epoch_item.repeat([5]).numpy()
for i in range(epoch_num.shape[0]):
    shuttle_fed_fnn_trend_data.append([epoch_num[i], shuttle_fed_fnn_trend_tsr.numpy()[0][i], shuttle_fed_fnn_trend_tsr.numpy()[1][i],
                               shuttle_fed_fnn_trend_tsr.numpy()[2][i], shuttle_fed_fnn_trend_tsr.numpy()[3][i]])
shuttle_fed_fnn_trend = DataFrame(shuttle_fed_fnn_trend_data, columns=["epoch", "00%", '10%', "20%", "30%"])

robot_trend_data = []
for i in range(noise.shape[0]):
    robot_trend_data.append([noise[i], dfnn_homo[5][i], dfnn_hetero[5][i], fed_dnn_ccvr_hetero[5][i],
                           fed_dnn_hetero[5][i], fed_dnn_homo[5][i], fed_fnn[5][i]])
robot_trend = DataFrame(robot_trend_data, columns=["uncertainty_level", "DFNN (+)", "DFNN (*)", 'FedDNN+CCVR+MOON (*)', "FedDNN (*)", "FedDNN (+)",
                               "FedFNN (*)"])
robot_fed_fnn_trend_data = []
robot_fed_fnn_trend_tsr = fed_fnn_trend_dict['robot']
epoch_item = torch.arange(int(robot_fed_fnn_trend_tsr.shape[1]/5)) + 1
epoch_num = epoch_item.repeat([5]).numpy()
for i in range(epoch_num.shape[0]):
    robot_fed_fnn_trend_data.append([epoch_num[i], robot_fed_fnn_trend_tsr.numpy()[0][i], robot_fed_fnn_trend_tsr.numpy()[1][i],
                               robot_fed_fnn_trend_tsr.numpy()[2][i], robot_fed_fnn_trend_tsr.numpy()[3][i]])
robot_fed_fnn_trend = DataFrame(robot_fed_fnn_trend_data, columns=["epoch", "00%", '10%', "20%", "30%"])

wifi_trend_data = []
for i in range(noise.shape[0]):
    wifi_trend_data.append([noise[i], dfnn_homo[6][i], dfnn_hetero[6][i], fed_dnn_ccvr_hetero[6][i],
                           fed_dnn_hetero[6][i], fed_dnn_homo[6][i], fed_fnn[6][i]])
wifi_trend = DataFrame(wifi_trend_data, columns=["uncertainty_level", "DFNN (+)", "DFNN (*)", 'FedDNN+CCVR+MOON (*)', "FedDNN (*)", "FedDNN (+)",
                               "FedFNN (*)"])
wifi_fed_fnn_trend_data = []
wifi_fed_fnn_trend_tsr = fed_fnn_trend_dict['wifi']
epoch_item = torch.arange(int(wifi_fed_fnn_trend_tsr.shape[1]/5)) + 1
epoch_num = epoch_item.repeat([5]).numpy()
for i in range(epoch_num.shape[0]):
    wifi_fed_fnn_trend_data.append([epoch_num[i], wifi_fed_fnn_trend_tsr.numpy()[0][i], wifi_fed_fnn_trend_tsr.numpy()[1][i],
                               wifi_fed_fnn_trend_tsr.numpy()[2][i], wifi_fed_fnn_trend_tsr.numpy()[3][i]])
wifi_fed_fnn_trend = DataFrame(wifi_fed_fnn_trend_data, columns=["epoch", "00%", '10%', "20%", "30%"])

#================plot fed_fnn trend figure====================
uncertainty_level_list = ["00%", '10%', "20%", "30%"]
plt.rcParams['figure.figsize']=[20, 12]
plt.subplots_adjust(wspace=0.3, hspace=0.4)
ax1 = plt.subplot(231)
ax1.legend(fontsize=15, loc='upper center', bbox_to_anchor=(2.35, 1.35), ncol=4, labels=uncertainty_level_list)

for i in range(len(uncertainty_level_list)):
    a = uncertainty_level_list[i]
    c = colors1[i]
    sns.lineplot(x="epoch", y=a, data=sdd_fed_fnn_trend, color=c, ci='sd', label=a)
# ax1.set_xlim((0, 51))
# ax1.set_ylim((0.77, 0.95))
ax1.set_ylabel('Test Accuracy',fontsize=18)
ax1.set_xlabel('Epoch',fontsize=18)
plt.yticks(size=16)
plt.xticks(size=16)
# ax1.locator_params('y',nbins=5)
# ax1.locator_params('x',nbins=5)
ax1.legend_.remove()
ax1.set_title('SDD', size=19)

ax2 = plt.subplot(232)
for i in range(len(uncertainty_level_list)):
    a = uncertainty_level_list[i]
    c = colors1[i]
    sns.lineplot(x="epoch", y=a, data=gsad_fed_fnn_trend, color=c, ci='sd', label=a)
# ax2.set_xlim((0, 250))
# ax2.set_ylim((0.77, 0.95))
ax2.set_ylabel('Test Accuracy',fontsize=18)
ax2.set_xlabel('Epoch',fontsize=18)
plt.yticks(size=16)
plt.xticks(size=16)
# ax2.locator_params('y',nbins=5)
# ax2.locator_params('x',nbins=5)
ax2.legend_.remove()
ax2.set_title('GSAD', size=19)

# ax3 = plt.subplot(243)
# for i in range(len(uncertainty_level_list)):
#     a = uncertainty_level_list[i]
#     c = colors1[i]
#     sns.lineplot(x="epoch", y=a, data=meterd_fed_fnn_trend, color=c, ci='sd', label=a)
# # ax3.set_xlim((0, 250))
# # ax3.set_ylim((0.77, 0.95))
# ax3.set_ylabel('Test Accuracy',fontsize=18)
# ax3.set_xlabel('Epoch',fontsize=18)
# # ax3.locator_params('y',nbins=5)
# # ax3.locator_params('x',nbins=5)
# ax3.legend_.remove()
# ax3.set_title('FM')

# ax4 = plt.subplot(244)
# for i in range(len(uncertainty_level_list)):
#     a = uncertainty_level_list[i]
#     c = colors1[i]
#     sns.lineplot(x="epoch", y=a, data=wine_fed_fnn_trend, color=c, ci='sd', label=a)
# # ax4.set_xlim((0, 250))
# # ax4.set_ylim((0.77, 0.95))
# ax4.set_ylabel('Test Accuracy',fontsize=18)
# ax4.set_xlabel('Epoch',fontsize=18)
# # ax4.locator_params('y',nbins=5)
# # ax4.locator_params('x',nbins=5)
# ax4.legend_.remove()
# ax4.set_title('WD')

ax5 = plt.subplot(233)
for i in range(len(uncertainty_level_list)):
    a = uncertainty_level_list[i]
    c = colors1[i]
    sns.lineplot(x="epoch", y=a, data=magic_fed_fnn_trend, color=c, ci='sd', label=a)
# ax5.set_xlim((0, 250))
# ax5.set_ylim((0.77, 0.95))
ax5.set_ylabel('Test Accuracy',fontsize=18)
ax5.set_xlabel('Epoch',fontsize=18)
plt.yticks(size=16)
plt.xticks(size=16)
# ax5.locator_params('y',nbins=5)
# ax5.locator_params('x',nbins=5)
ax5.legend_.remove()
ax5.set_title('MGT', size=19)

ax6 = plt.subplot(234)
for i in range(len(uncertainty_level_list)):
    a = uncertainty_level_list[i]
    c = colors1[i]
    sns.lineplot(x="epoch", y=a, data=shuttle_fed_fnn_trend, color=c, ci='sd', label=a)
# ax6.set_xlim((0, 250))
# ax6.set_ylim((0.77, 0.95))
ax6.set_ylabel('Test Accuracy',fontsize=18)
ax6.set_xlabel('Epoch',fontsize=18)
plt.yticks(size=16)
plt.xticks(size=16)
# ax6.locator_params('y',nbins=5)
# ax6.locator_params('x',nbins=5)
ax6.legend_.remove()
ax6.set_title('SC', size=19)

ax7 = plt.subplot(235)
for i in range(len(uncertainty_level_list)):
    a = uncertainty_level_list[i]
    c = colors1[i]
    sns.lineplot(x="epoch", y=a, data=robot_fed_fnn_trend, color=c, ci='sd', label=a)
# ax7.set_xlim((0, 250))
# ax7.set_ylim((0.77, 0.95))
ax7.set_ylabel('Test Accuracy',fontsize=18)
ax7.set_xlabel('Epoch',fontsize=18)
plt.yticks(size=16)
plt.xticks(size=16)
# ax7.locator_params('y',nbins=5)
# ax7.locator_params('x',nbins=5)
ax7.legend_.remove()
ax7.set_title('WFRN', size=19)

ax8 = plt.subplot(236)
for i in range(len(uncertainty_level_list)):
    a = uncertainty_level_list[i]
    c = colors1[i]
    sns.lineplot(x="epoch", y=a, data=wifi_fed_fnn_trend, color=c, ci='sd', label=a)
# ax8.set_xlim((0, 250))
# ax8.set_ylim((0.77, 0.95))
ax8.set_ylabel('Test Accuracy',fontsize=18)
ax8.set_xlabel('Epoch',fontsize=18)
plt.yticks(size=16)
plt.xticks(size=16)
# ax8.locator_params('y',nbins=5)
# ax8.locator_params('x',nbins=5)
ax8.legend_.remove()
ax8.set_title('WIL', size=19)
ax1.legend(fontsize=20, loc='upper center', bbox_to_anchor=(1.75, 1.40), ncol=4)
plt.savefig(f"./results/fed_fnn_trend.pdf",bbox_inches='tight')
