import os
import argparse
from tqdm import tqdm
from datetime import datetime

import torch
import torchvision
import torch.cuda.amp as amp
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import snetx
from snetx import utils
import snetx.snn.algorithm as snnalgo

import dvs_datasets.vision as dvsds
import datasets.vision as visionds
from snetx.dataset import vision as snnvds

import snetx.models
from snetx.models import vgg, ms_resnet, sew_resnet
from snetx.snn import neuron
import nwarmup


def amp_train_dvs(net, dataloader, optimizer, scaler, args):
    net.train()
    running_loss = sumup = 0.0
    correct = 0.
    
    for i, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.float().to(0, non_blocking=True)
        labels = labels.to(0, non_blocking=True)
            
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
def validate_dvs(net, dataloader, device, args,):
    net.eval()
    correct = 0.
    sumup = 0.
    for inputs, labels in dataloader:
        inputs = inputs.float().to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        out = net(inputs).mean(dim=1)
            
        sumup += inputs.shape[0]
        correct += out.argmax(dim=1).eq(labels).sum().item()
        
    net.train()
    return correct, sumup

def amp_train_static(net, dataloader, optimizer, scaler, args):
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
    
    if args.dataset == 'CIFAR10':
        tr_data, ts_data = snnvds.cifar10_dataset(args.data_dir, args.batch_size1, args.batch_size2)
        dvs = False
    elif args.dataset == 'CIFAR100':
        normalize = torchvision.transforms.Normalize(
            mean=[0.507, 0.487, 0.441],
            std=[0.267, 0.256, 0.276],
        )
        
        transform_train = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(0.5),
            torchvision.transforms.ToTensor(),
            snetx.utils.Cutout(1, 16),
            normalize
        ])
        
        transform_test = torchvision.transforms.Compose([
        #     transforms.RandomCrop(32, padding=4),
        #     transforms.RandomHorizontalFlip(0.5),
            torchvision.transforms.ToTensor(),
            normalize
        ])
        tr_data, ts_data = snnvds.cifar100_dataset(args.data_dir, args.batch_size1, args.batch_size2, [transform_train, transform_test])
        dvs = False
    elif args.dataset == 'CIFAR10DVS':
        tr_data, ts_data = dvsds.cifar10dvs_dataset(args.data_dir, args.batch_size1, args.batch_size2, args.T)
        dvs = True
    elif args.dataset == 'DVSGesture128':
        tr_data, ts_data = dvsds.dvs128gesture_dataset(args.data_dir, args.batch_size1, args.batch_size2, args.T)
        dvs = True
    else:
        raise ValueError(f'{args.dataset} not supported.')
    
    neuron_cfg = {
        # 'alpha': nwarmup.ConsineAnnealingWarmup(args.base, args.bound, args.T_max2), 
        'alpha': nwarmup.PolynormialWarmup(args.base, args.bound, args.T_max2),
        'tau': args.tau,
    }
    if 'resnet' not in args.arch:
        if 'DVS' in args.dataset:
            in_channels = 2
        else:
            in_channels = 3
        net = vgg.__dict__[args.arch](neuron.LIF, neuron_cfg, in_channels, args.drop, BN, num_classes=args.num_classes).to(device)
    elif 'sew' in args.arch:
        if 'DVS' not in args.dataset:
            feature = snetx.models.sew_resnet.cifar10_feature
        else:
            feature = snetx.models.sew_resnet.cifar10dvs_feature
        net = snetx.models.sew_resnet.__dict__[args.arch[3:]](neuron.LIF, neuron_cfg, num_classes=args.num_classes, feature=feature, norm_layer=BN).to(device)
    elif 'ms' in args.arch:
        if 'DVS' not in args.dataset:
            feature = snetx.models.ms_resnet.cifar10_feature
        else:
            feature = snetx.models.ms_resnet.cifar10dvs_feature
        net = snetx.models.ms_resnet.__dict__[args.arch[2:]]('A', neuron.LIF, neuron_cfg, num_classes=args.num_classes, feature=feature, norm_layer=BN).to(device)
    else:
        raise ValueError(f'{args.arch} is not supported.')
    
    print(net)
    net.train()
    
    if args.amp:
        scaler = amp.GradScaler()
    else:
        scaler = None
    
    if args.debug:
        writer = None
    else:
        writer = SummaryWriter(log_dir=f'{args.logs_dir}/{args.dataset}/{args.arch}/{args.T}/{args.mode}')
    
    if args.optim == 'AdamW':
        optimizer = torch.optim.AdamW(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=0.9)
        # optimizer = torch.optim.SGD(net.parameters(), lr=args.learning_rate / args.T, weight_decay=args.weight_decay, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max1, eta_min=args.eta_min)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.98)
    
    max_acc = 0.
    for e in range(args.num_epochs):
        if args.verbose:
            dataloader = tqdm(tr_data)
        else:
            dataloader = tqdm(tr_data)
        
        if dvs:
            pass
            if args.amp:
                correct, sumup, loss = amp_train_dvs(net, dataloader, optimizer, scaler, args)
            else:
                correct, sumup, loss = amp_train_dvs(net, dataloader, optimizer, args)
            correct, sumup = validate_dvs(net, ts_data, device, args)
        else:
            if args.amp:
                correct, sumup, loss = amp_train_static(net, dataloader, optimizer, scaler, args)
            else:
                correct, sumup, loss = amp_train_static(net, dataloader, optimizer, args)
            correct, sumup = validate_static(net, ts_data, device, args)
          
        if not args.debug:      
            writer.add_scalar('Loss', loss, e)
            writer.add_scalar('Acc', correct / sumup, e)
        
        max_acc = max(max_acc, correct)

        print('epoch: ', e, f'loss: {loss:.4f}, Acc: {(correct / sumup) * 100:.2f}%, Best: {(max_acc / sumup) * 100:.2f}%')  
        print(scheduler.get_last_lr(), neuron_cfg['alpha'].get_last_alpha())
        
        scheduler.step()
        neuron_cfg['alpha'].step()
    
    if not args.debug: writer.close()

def main():
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size1', type=int, default=32, help='batch size for single device.')
    parser.add_argument('--batch_size2', type=int, default=256, help='test batch size.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for gradient descent.')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='penal term parameter for model weight.')
    parser.add_argument('--optim', type=str, default='SGD', help='AdamW, SGD')
    
    parser.add_argument('--tau', type=float, default=2., help='')
    parser.add_argument('--reset', type=str, default='hard', help='')
    parser.add_argument('--T', type=int, default=6, help='snn simulate time step.')
    parser.add_argument('--num_epochs', type=int, default=1, help='max epochs for train process.')
    parser.add_argument('--T_max1', type=int, default=200, help='schedule period for consine annealing lr scheduler.')
    parser.add_argument('--T_max2', type=int, default=200, help='schedule period for consine annealing neuron warmup.')
    parser.add_argument('--eta_min', type=float, default=0.)
    parser.add_argument('--base', type=float, default=1., help='')
    parser.add_argument('--bound', type=float, default=1., help='')
    parser.add_argument('--drop', type=float, default=0.2, help='')
    parser.add_argument('--mode', type=str, default='s', help='')
    
    
    parser.add_argument('--dataset', type=str, default='DVSGesture128', help='')
    parser.add_argument('--data_dir', type=str, default='./data_dir/DVS128Gesture', help='data directory.')
    # parser.add_argument('--dataset', type=str, default='CIFAR10DVS', help='')
    # parser.add_argument('--data_dir', type=str, default='../dataset/CIFAR10DVS', help='data directory.')
    # parser.add_argument('--dataset', type=str, default='CIFAR10', help='')
    # parser.add_argument('--data_dir', type=str, default='./data_dir', help='data directory.')
    parser.add_argument('--logs_dir', type=str, default='./LOGS/TestMemBPTT', help='logs directory.')
    
    parser.add_argument('--print_intv', type=int, default=50, 
                        help='train steps interval to print train mesasges: show messages after each {intv} batches.')
    parser.add_argument('--amp', '-A', action='store_true')

    parser.add_argument('--verbose', '-V', action='store_true', 
                        help='whether to display training progress in the master process.')
    parser.add_argument('--arch', type=str, default='msresnet18', 
                        help='network architecture.'
                             'sewewsnet{}: sew resnet 18, 34, 50, 101'
                             'msresnet{}: ms resnet 18, 34, 50, 101'
                             'vgg:'
                        )
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--debug', '-D', action='store_true')
    cmd_args = parser.parse_args()
    
    execuate(torch.device(cmd_args.device), cmd_args)
    

if __name__ == '__main__':
    main()