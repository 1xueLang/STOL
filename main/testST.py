import os
import argparse
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.cuda.amp as amp
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import snetx
from snetx import utils
import snetx.snn.algorithm as snnalgo
from snetx.models import ms_resnet#, vgg
from snetx.snn import neuron

import dvs_datasets.vision as dvsds
import datasets.vision as visionds
from snetx.dataset import vision as snnvds

import vgg
import resnet
import nwarmup
import training
import online

def neuronOnLine_reset(modules):
    for m in modules:
        if isinstance(m, STOLO):
            m.u = 0.
            
def amp_train_static(net, dataloader, optimizer, scaler, args):
    net.train()
    running_loss = sumup = 0.0
    correct = 0.
    
    for i, (inputs, labels) in enumerate(dataloader):
        
        inputs = inputs.float().to(0, non_blocking=True)
        labels = labels.to(0, non_blocking=True)
        
        neuronOnLine_reset(net.modules())
        sumout = 0.
            
        for t in range(args.T):
            with amp.autocast():
                out = net(inputs)
                loss = F.cross_entropy(out, labels) # + 5e-2 * F.mse_loss(out, F.one_hot(labels, args.num_classes).to(out))
                
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
            sumout += out
    
        running_loss += loss.item()
        sumup += inputs.shape[0]
        
        correct += sumout.argmax(dim=1).eq(labels).sum().item()
        
        if (1 + i) % args.print_intv == 0:
            print(f'  Loss: {loss.item():.4f}, Acc: {(correct / sumup) * 100:.2f}%')
        
    return correct, sumup, float(running_loss / len(dataloader))

@torch.no_grad()
def STOLO_validate_static(net, dataloader, device, args,):
    net.eval()
    correct = 0.
    sumup = 0.
    for inputs, labels in dataloader:
        inputs = inputs.float().to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        sumout = 0.
        neuronOnLine_reset(net.modules())
        
        for t in range(args.T):
            out = net(inputs)
            sumout += out
            
        sumup += inputs.shape[0]
        correct += sumout.argmax(dim=1).eq(labels).sum().item()
        
    net.train()
    return correct, sumup

class STOLO(nn.Module):
    def __init__(self, tau=2., sg=snnalgo.PiecewiseQuadratic):
        super().__init__()
        self.tau = tau
        self.spiking = sg.apply
        self.u = 0.
    
    def forward(self, x):
        u = self.u / self.tau + x
        s = self.spiking(u - 1., 1.)
        self.u = (u * (1. - s)).detach()
        return s


class VGG11(nn.Module):
    def __init__(self, init_weights=True, num_classes=10, dropout=0.2):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            STOLO(),
            nn.AvgPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            STOLO(),
            nn.AvgPool2d(2, 2),
            
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            STOLO(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            STOLO(),
            nn.AvgPool2d(2, 2),
            
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            STOLO(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            STOLO(),
            nn.AvgPool2d(2, 2),
            
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            STOLO(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            STOLO(),
            nn.AvgPool2d(2, 2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(512 * 7 * 7, num_classes),
        )
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x.flatten(1))
        return x

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
        self.bn = torch.nn.BatchNorm2d(in_channels)
        
    def forward(self, x):
        out = []
        for i in range(x.shape[1]):
            out.append(self.bn(x[:, i]))
        return torch.stack(out, dim=1)

def execuate(device, args):
    
    tr_data, ts_data = snnvds.cifar10_dataset(args.data_dir, args.batch_size1, args.batch_size2)
    dvs = False
    
    neuron_cfg = {
        'alpha': nwarmup.PolynormialWarmup(1., 1., 200),
        'tau': 2.,
        # 'sg': snnalgo.PiecewiseQuadratic,
    }
    
    # net_bptt = vgg.__dict__['vgg11_bn'](neuron.LIF, neuron_cfg, in_channels=3, num_classes=10, dropout=0.2, norm_layer=BN).to(device)
    neuron_cfg['mode'] = 's'
    net_stol_s = vgg.__dict__['vgg11_bn'](online.neuronOnLine, neuron_cfg, in_channels=3, num_classes=10, dropout=0.2, norm_layer=nn.BatchNorm2d).to(device)
    neuron_cfg['mode'] = 't'
    net_stol_t = vgg.__dict__['vgg11_bn'](online.neuronOnLine, neuron_cfg, in_channels=3, num_classes=10, dropout=0.2, norm_layer=nn.BatchNorm2d).to(device)
    # neuron_cfg['mode'] = 'o'
    net_stol_o = VGG11(dropout=0.2).to(device)

    # net = net_bptt
    net = net_stol_o
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, weight_decay=5e-5, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    
    # writer = SummaryWriter(log_dir=f'LOGS/TestST/bptt')
    
    scaler = amp.GradScaler()
    
    max_acc = 0.
    for e in range(2):
    # for e in range(args.num_epochs):
        if args.verbose:
            dataloader = tqdm(tr_data)
        else:
            dataloader = tqdm(tr_data)
        
        correct, sumup, loss = amp_train_static(net, dataloader, optimizer, scaler, args)
        correct, sumup = STOLO_validate_static(net, ts_data, device, args)
        # correct, sumup, loss = training.amp_train_static(net, dataloader, optimizer, scaler, args)
        # correct, sumup = training.validate_static(net, ts_data, device, args)
        # correct, sumup, loss = train_static(net, dataloader, optimizer, args, scaler)
        # correct, sumup = validate_static(net, ts_data, device, args)
        
        max_acc = max(max_acc, correct)

        print('epoch: ', e, f'loss: {loss:.4f}, Acc: {(correct / sumup) * 100:.2f}%, Best: {(max_acc / sumup) * 100:.2f}%')  
        print(scheduler.get_last_lr(), neuron_cfg['alpha'].get_last_alpha())
        
        # writer.add_scalar('Loss', loss, e)
        # writer.add_scalar('Acc', correct / sumup, e)
        
        scheduler.step()
        neuron_cfg['alpha'].step()
    
    # writer.close()

def main():
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size1', type=int, default=128, help='batch size for single device.')
    parser.add_argument('--batch_size2', type=int, default=256, help='test batch size.')
    
    parser.add_argument('--T', type=int, default=32, help='snn simulate time step.')
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
    # T: 2, 4, 8, 16, 32
    # BPTT   3823MB 1m21s 18.01% 2431 3081 4331 6747 12029
    # STOL-O 2287MB 1m26s 36.96% 2287 2303 2303 2303 2303
    # STOL-T 2519MB 1m40s 37.25% 2483 2499 2515 2515 2515
    # STOL-S 2519MB 1m46s 43.59% 2483 2499 2515 2515 2515