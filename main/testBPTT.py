import os
import argparse
from tqdm import tqdm
from datetime import datetime

import torch
import torch.cuda.amp as amp
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import snetx
from snetx import utils
import snetx.snn.algorithm as snnalgo
from snetx.models import ms_resnet
from snetx.snn import neuron

import dvs_datasets.vision as dvsds
import datasets.vision as visionds
from snetx.dataset import vision as snnvds

# import vgg
# import resnet
import nwarmup

# import online

def train_static(net, dataloader, optimizer, args, scaler):
    net.train()
    running_loss = sumup = 0.0
    correct = 0.
    
    for i, (inputs, labels) in enumerate(dataloader):
        
        inputs = inputs.float().to(0, non_blocking=True)
        labels = labels.to(0, non_blocking=True)
        inputs = snnalgo.temporal_repeat(inputs, args.T)
        
        with amp.autocast():
            out = net(inputs)
            loss = 0.
            for t in range(args.T):
                loss += F.cross_entropy(out[:, t], labels) 

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    
        running_loss += loss.item()
        sumup += inputs.shape[0]
        
        correct += out.mean(dim=1).argmax(dim=1).eq(labels).sum().item()
        
        if (1 + i) % args.print_intv == 0:
            print(f'  Loss: {loss.item():.4f}, Acc: {(correct / sumup) * 100:.2f}%')
        
    return correct, sumup, float(running_loss / len(dataloader))

@torch.no_grad()
def validate_static(net, dataloader, device, args,):
    net.eval()
    correct = 0.
    sumup = 0.
    for inputs, labels in dataloader:
        inputs = inputs.float().to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        inputs = snnalgo.temporal_repeat(inputs, args.T)
        
        out = net(inputs).mean(dim=1)
            
        sumup += inputs.shape[0]
        correct += out.argmax(dim=1).eq(labels).sum().item()
        
    net.train()
    return correct, sumup

class BN(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # self.bn = torch.nn.BatchNorm2d(in_channels)
        
    def forward(self, x):
        # out = []
        # for i in range(x.shape[1]):
        #     out.append(self.bn(x[:, i]))
        # return torch.stack(out, dim=1)
        return x

def execuate(device, args):
    
    tr_data, ts_data = snnvds.cifar10_dataset(args.data_dir, args.batch_size1, args.batch_size2)
    dvs = False
    
    neuron_cfg = {
        'alpha': nwarmup.PolynormialWarmup(1., 1., 200),
        'tau': 2.,
        'sg': snnalgo.PiecewiseQuadratic,
    }
    
    feature = ms_resnet.cifar10_feature
    net_bptt = ms_resnet.__dict__['resnet18']('A', neuron.LIF, neuron_cfg, num_classes=10, feature=feature, norm_layer=BN).to(device)
    
    # feature = resnet.ms_resnet.cifar10_feature
    # neuron_cfg['mode'] = 's'
    # net_stol_s = resnet.ms_resnet.__dict__['resnet18']('A', online.neuronOnLine, neuron_cfg, num_classes=10, feature=feature).to(device)
    # neuron_cfg['mode'] = 't'
    # net_stol_t = resnet.ms_resnet.__dict__['resnet18']('A', online.neuronOnLine, neuron_cfg, num_classes=10, feature=feature).to(device)
    # neuron_cfg['mode'] = 'o'
    # net_stol_o = resnet.ms_resnet.__dict__['resnet18']('A', online.neuronOnLine, neuron_cfg, num_classes=10, feature=feature).to(device)

    optimizer = torch.optim.AdamW(net_bptt.parameters(), lr=0.001, weight_decay=5e-5)
    # optimizer = torch.optim.SGD(net_bptt.parameters(), lr=0.15, weight_decay=5e-5, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    
    writer = SummaryWriter(log_dir=f'LOGS/CIFAR-10/ms_resnet18/bptt')
    
    scaler = amp.GradScaler()
    
    max_acc = 0.
    for e in range(args.num_epochs):
        if args.verbose:
            dataloader = tqdm(tr_data)
        else:
            dataloader = tqdm(tr_data)
        
        correct, sumup, loss = train_static(net_bptt, dataloader, optimizer, args, scaler)
        correct, sumup = validate_static(net_bptt, ts_data, device, args)
        
        max_acc = max(max_acc, correct)

        print('epoch: ', e, f'loss: {loss:.4f}, Acc: {(correct / sumup) * 100:.2f}%, Best: {(max_acc / sumup) * 100:.2f}%')  
        print(scheduler.get_last_lr(), neuron_cfg['alpha'].get_last_alpha())
        
        writer.add_scalar('Loss', loss, e)
        writer.add_scalar('Acc', correct / sumup, e)
        
        scheduler.step()
        neuron_cfg['alpha'].step()
    writer.close()
        

def main():
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size1', type=int, default=128, help='batch size for single device.')
    parser.add_argument('--batch_size2', type=int, default=256, help='test batch size.')
    
    parser.add_argument('--T', type=int, default=6, help='snn simulate time step.')
    parser.add_argument('--num_epochs', type=int, default=200, help='max epochs for train process.')
    
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='')
    parser.add_argument('--data_dir', type=str, default='./data_dir', help='data directory.')
    
    parser.add_argument('--print_intv', type=int, default=50, 
                        help='train steps interval to print train mesasges: show messages after each {intv} batches.')
    parser.add_argument('--amp', '-A', action='store_true')

    parser.add_argument('--verbose', '-V', action='store_true', 
                        help='whether to display training progress in the master process.')
    parser.add_argument('--device', type=str, default='cuda')
    cmd_args = parser.parse_args()
    
    execuate(torch.device(cmd_args.device), cmd_args)
    

if __name__ == '__main__':
    main()