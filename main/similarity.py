# import torch
# import numpy as np
# import nwarmup
# import resnet
# import online
# import torch.nn.functional as F
# from snetx.models import ms_resnet
# from snetx.snn import neuron
# from snetx.dataset import vision as snnvds
# import snetx.snn.algorithm as snnalgo

# from tqdm import tqdm
# import load

# def bptt_backward(net, inputs, labels):
#     inputs = inputs.float().to(0, non_blocking=True)
#     labels = labels.to(0, non_blocking=True)
#     inputs = snnalgo.temporal_repeat(inputs, 6)
        
#     out = net(inputs)
#     loss = 0.
#     for t in range(6):
#         loss += F.cross_entropy(out[:, t], labels)
        
#     loss.backward()

# def stol_backward(net, inputs, labels):
#     inputs = inputs.float().to(0, non_blocking=True)
#     labels = labels.to(0, non_blocking=True)
        
#     online.neuronOnLine_reset(net.modules())

#     for t in range(6):
#         if t == 0:
#             histx = torch.zeros_like(inputs)
#         else:
#             histx = inputs
#         out = net((inputs, histx))[0]
#         loss = F.cross_entropy(out, labels)
#         loss.backward()


# def similarity(x1, x2):
#     # print(x1.shape, x2.shape)
#     return [torch.cosine_similarity(x1, x2).item(), torch.pairwise_distance(x1, x2).item()]

# class BN(torch.nn.Module):
#     def __init__(self, in_channels):
#         super().__init__()
#         self.bn = torch.nn.BatchNorm2d(in_channels)
        
#     def forward(self, x):
#         out = []
#         for i in range(x.shape[1]):
#             out.append(self.bn(x[:, i]))
#         return torch.stack(out, dim=1)

# device = 0

# neuron_cfg = {
#     'alpha': nwarmup.PolynormialWarmup(1., 1., 200),
#     'tau': 2.,
#     'sg': snnalgo.PiecewiseQuadratic,
# }

# tr_data, ts_data = snnvds.cifar10_dataset('./data_dir/', 64, 32)

# # x, y = next(iter(tr_data))

# feature = ms_resnet.cifar10_feature
# net_bptt = ms_resnet.__dict__['resnet18']('A', neuron.LIF, neuron_cfg, num_classes=10, feature=feature, norm_layer=BN).to(device)

# neuron_cfg = {
#     'alpha': nwarmup.PolynormialWarmup(1., 1., 200),
#     'tau': 2.,
# }

# feature = resnet.ms_resnet.cifar10_feature
# neuron_cfg['mode'] = 's'
# net_stol_s = resnet.ms_resnet.__dict__['resnet18']('A', online.neuronOnLine, neuron_cfg, num_classes=10, feature=feature).to(device)
# neuron_cfg['mode'] = 't'
# net_stol_t = resnet.ms_resnet.__dict__['resnet18']('A', online.neuronOnLine, neuron_cfg, num_classes=10, feature=feature).to(device)
# neuron_cfg['mode'] = 'o'
# net_stol_o = resnet.ms_resnet.__dict__['resnet18']('A', online.neuronOnLine, neuron_cfg, num_classes=10, feature=feature).to(device)

# optimizer1 = torch.optim.SGD(net_bptt.parameters(), lr=0.05, weight_decay=5e-5, momentum=0.9)
# optimizer2 = torch.optim.SGD(net_stol_s.parameters(), lr=0.05, weight_decay=5e-5, momentum=0.9)
# optimizer3 = torch.optim.SGD(net_stol_t.parameters(), lr=0.05, weight_decay=5e-5, momentum=0.9)
# optimizer4 = torch.optim.SGD(net_stol_o.parameters(), lr=0.05, weight_decay=5e-5, momentum=0.9)

# sim1 = []
# sim2 = []
# sim3 = []

# for e in [0, 49, 99, 199]:
#     sim1.append([])
#     sim2.append([])
#     sim3.append([])
#     state = torch.load(f'storage/epoch{e + 1}.pth')
    
#     for x, y in tqdm(tr_data):
#         net_bptt.load_state_dict(state)
#         load.load_from_bptt(net_stol_s, state)
#         load.load_from_bptt(net_stol_t, state)
#         load.load_from_bptt(net_stol_o, state)
#         sim1[-1].append([])
#         sim2[-1].append([])
#         sim3[-1].append([])
#         optimizer1.zero_grad()
#         optimizer2.zero_grad()
#         optimizer3.zero_grad()
#         optimizer4.zero_grad()
        
#         bptt_backward(net_bptt, x, y)
#         stol_backward(net_stol_s, x, y)
#         stol_backward(net_stol_t, x, y)
#         stol_backward(net_stol_o, x, y)
        
#         for p1, p2 in zip(net_bptt.named_parameters(), net_stol_s.named_parameters()):
#             if 'weight' in p1[0] and 'bn' not in p1[0] and 'downsample' not in p1[0]:
#                 sim1[-1][-1].append(similarity(p1[1].grad.flatten().view(1, -1), p2[1].grad.flatten().view(1, -1)))
        
#         for p1, p2 in zip(net_bptt.named_parameters(), net_stol_t.named_parameters()):
#             if 'weight' in p1[0] and 'bn' not in p1[0] and 'downsample' not in p1[0]:
#                 sim2[-1][-1].append(similarity(p1[1].grad.flatten().view(1, -1), p2[1].grad.flatten().view(1, -1)))
        
#         for p1, p2 in zip(net_bptt.named_parameters(), net_stol_o.named_parameters()):
#             if 'weight' in p1[0] and 'bn' not in p1[0] and 'downsample' not in p1[0]:
#                 sim3[-1][-1].append(similarity(p1[1].grad.flatten().view(1, -1), p2[1].grad.flatten().view(1, -1)))


# sim1 = np.array(sim1)
# mean1 = np.mean(sim1, axis=1)
# var1 = np.var(sim1,axis=1)
# sim2 = np.array(sim2)
# mean2 = np.mean(sim2, axis=1)
# var2 = np.var(sim2,axis=1)
# sim3 = np.array(sim3)
# mean3 = np.mean(sim3, axis=1)
# var3 = np.var(sim3,axis=1)

# np.save('storage/sim1.npy', sim1)
# np.save('storage/sim2.npy', sim2)
# np.save('storage/sim3.npy', sim3)

# print(mean1.shape, var1.shape)
# print(var1)

# from matplotlib import pyplot as plt 

# c = ["233d4d","fe7f2d","fcca46","a1c181","619b8a", "e63946", "a8dadc"]
# c = ['#' + cl for cl in c]
# np.random.shuffle(c)

# j = 2
# for e, epoch in zip(range(4), [1, 50, 100, 200]):
#     fig, ax = axs[i].subplots()

#     axs[i].errorbar(np.arange(18) + 1, mean1[e, :, 0], yerr=var1[e, :, 0], label='cos-S', color=c[0])
#     axs[i].errorbar(np.arange(18) + 1, mean2[e, :, 0], yerr=var2[e, :, 0], label='cos-T', color=c[1])
#     axs[i].errorbar(np.arange(18) + 1, mean3[e, :, 0], yerr=var3[e, :, 0], label='cos-O', color=c[2])

#     axs[i].errorbar(np.arange(18) + 1, mean1[e, :, 1], yerr=var1[e, :, 1], label='L2-S', color=c[0])
#     axs[i].errorbar(np.arange(18) + 1, mean2[e, :, 1], yerr=var2[e, :, 1], label='L2-T', color=c[1])
#     axs[i].errorbar(np.arange(18) + 1, mean3[e, :, 1], yerr=var3[e, :, 1], label='L2-O', color=c[2])

#     # axs[i].plot(np.arange(18) + 1, sim1[j, e, :, 0], linewidth=2.0, label='cos-S', color=c[0])
#     # axs[i].plot(np.arange(18) + 1, sim2[j, e, :, 0], linewidth=2.0, label='cos-T', color=c[1])
#     # axs[i].plot(np.arange(18) + 1, sim3[j, e, :, 0], linewidth=2.0, label='cos-O', color=c[2])

#     # axs[i].plot(np.arange(18) + 1, sim1[j, e, :, 1], linewidth=2.0, label='L2-S', color=c[3])
#     # axs[i].plot(np.arange(18) + 1, sim2[j, e, :, 1], linewidth=2.0, label='L2-T', color=c[4])
#     # axs[i].plot(np.arange(18) + 1, sim3[j, e, :, 1], linewidth=2.0, label='L2-O', color=c[5])
    
#     axs[i].legend()
#     axs[i].grid()
#     axs[i].savefig(f'Figure/sim{epoch}.png')
    
    
###################################

# import seaborn as sns
# import numpy as np
# from pandas import DataFrame
# from matplotlib import rcParams

# sim1 = np.load('storage/sim1.npy')
# sim2 = np.load('storage/sim2.npy')
# sim3 = np.load('storage/sim3.npy')
# print(sim1.shape)

# labels = ['CS-0', 'CS-1', 'CS-2', 'ED-0', 'ED-1', 'ED-2']

# print(sim1.shape)

# sims = []

# for i in range(4):
#     sims.append([])
#     for j in range(781):
#         for k in range(18):
#             _ = [labels[0], k + 1, sim1[i, j, k, 0]]
#             sims[-1].append(_)
#             _ = [labels[1], k + 1, sim2[i, j, k, 0]]
#             sims[-1].append(_)
#             _ = [labels[2], k + 1, sim3[i, j, k, 0]]
#             sims[-1].append(_)
#             _ = [labels[3], k + 1, sim1[i, j, k, 1]]
#             sims[-1].append(_)
#             _ = [labels[4], k + 1, sim2[i, j, k, 1]]
#             sims[-1].append(_)
#             _ = [labels[5], k + 1, sim3[i, j, k, 1]]
#             sims[-1].append(_)

# sns.set_style("whitegrid")
# sns.color_palette("Set2")
# c = [["ef476f","ffd166","06d6a0"], ['fa7f6f', 'ffbe7a', '8ecfc9']]

# for i, epoch in zip(range(4), [1, 50, 100, 200]):            
#     df = DataFrame(sims[i])
#     df.columns = ['Lines', 'Layer', 'Consie Similarity(CS) / Euclidean Distance(ED)']
#     fig = sns.relplot(
#         data=df, x='Layer', y='Consie Similarity(CS) / Euclidean Distance(ED)', kind="line", hue='Lines', errorbar='sd', legend='brief',
#     )
#     fig.savefig(f"Figure/epoch{epoch}.png")



# import numpy as np
# from pandas import DataFrame
# from matplotlib import rcParams
# from matplotlib import pyplot as plt

# rcParams.update({'font.size': 16})
# plt.rcParams["figure.figsize"] = (20, 4)

# sim1 = np.load('storage/sim1.npy')
# sim2 = np.load('storage/sim2.npy')
# sim3 = np.load('storage/sim3.npy')
# print(sim1.shape)
# mean1 = sim1.mean(axis=1)
# var1 = sim1.std(axis=1)
# min1 = mean1 - var1
# max1 = mean1 + var1
# # min1 = sim1.min(axis=1)
# # max1 = sim1.max(axis=1)
# mean2 = sim2.mean(axis=1)
# var2 = sim2.std(axis=1)
# min2 = mean2 - var2
# max2 = mean2 + var2
# # min2 = sim2.min(axis=1)
# # max2 = sim2.max(axis=1)
# mean3 = sim3.mean(axis=1)
# var3 = sim3.std(axis=1)
# min3 = mean3 - var3
# max3 = mean3 + var3
# # min3 = sim3.min(axis=1)
# # max3 = sim3.max(axis=1)

# labels = ['CS-0', 'CS-1', 'CS-2', 'ED-0', 'ED-1', 'ED-2']

# print(sim1.shape)

# sims = []
# x = np.arange(18) + 1

# # c = ["ef476f","ffd166","06d6a0", 'fa7f6f', 'ffbe7a', '8ecfc9']
# c = ["f47068","ffb3ae","fff4f1","1697a6","0e606b","ffc24b"]
# c = ['#' + cl for cl in c]
# # np.random.shuffle(c)

# fig, axs = plt.subplots(1, 4, sharey=True)

# for i, e in zip(range(4), [1, 50, 100, 200]):
#     axs[i].plot(x, mean1[i, :, 0], label=labels[0], color=c[0], marker='1')
#     axs[i].fill_between(x, min1[i, :, 0], max1[i, :, 0], alpha=0.2, color=c[0])

#     axs[i].plot(x, mean2[i, :, 0], label=labels[1], color=c[1], marker='1')
#     axs[i].fill_between(x, min2[i, :, 0], max2[i, :, 0], alpha=0.2, color=c[1])

#     axs[i].plot(x, mean3[i, :, 0], label=labels[2], color=c[2], marker='1')
#     axs[i].fill_between(x, min3[i, :, 0], max3[i, :, 0], alpha=0.2, color=c[2])

#     axs[i].plot(x, mean1[i, :, 1], label=labels[3], color=c[3], marker='1')
#     axs[i].fill_between(x, min1[i, :, 1], max1[i, :, 1], alpha=0.2, color=c[3])

#     axs[i].plot(x, mean2[i, :, 1], label=labels[4], color=c[4], marker='1')
#     axs[i].fill_between(x, min2[i, :, 1], max2[i, :, 1], alpha=0.2, color=c[4])

#     axs[i].plot(x, mean3[i, :, 1], label=labels[5], color=c[5], marker='1')
#     axs[i].fill_between(x, min3[i, :, 1], max3[i, :, 1], alpha=0.2, color=c[5])
    
#     axs[i].set_xticks(range(2, 18, 2))
#     axs[i].set_xlabel(f'Epoch = {e}')
#     axs[i].grid()
    
    
# axs[0].legend()

# fig.subplots_adjust(left=0.055, wspace=0)
# # plt.xlabel('Layer')
# fig.supylabel('(Consie, L2) Distance')
# plt.savefig(f'Figure/epoch.png', bbox_inches='tight', pad_inches=0.025)
# plt.savefig(f'Figure/pdfs/epoch.pdf', bbox_inches='tight', pad_inches=0.025)

import numpy as np
import pandas as pd


sim1 = np.load('storage/sim1.npy')
sim2 = np.load('storage/sim2.npy')
sim3 = np.load('storage/sim3.npy')

mean1 = sim1.mean(axis=1)
sd1 = sim1.std(axis=1)
mean2 = sim2.mean(axis=1)
sd2 = sim2.std(axis=1)
mean3 = sim3.mean(axis=1)
sd3 = sim3.std(axis=1)

df = pd.DataFrame(columns=['Mean1', 'Sd1', 'Mean2', 'Sd2', 'Layer', 'Epoch', 'Method'])

def make_df(df, mean_, sd_, m):
    es = [1, 50, 100, 200]
    for i in range(4):
        for j in range(18):
            df.loc[len(df.index)] = [mean_[i, j, 0], sd_[i, j, 0], mean_[i, j, 1], sd_[i, j, 1], j + 1, es[i], m]
            
make_df(df, mean1, sd1, 'STOL-S')
make_df(df, mean2, sd2, 'STOL-T')
make_df(df, mean3, sd3, 'STOL-O')

df.to_csv('storage/grad_sim.csv', index=False)
