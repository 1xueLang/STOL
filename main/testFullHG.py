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

from snetx.dataset import vision as snnvds

# import vgg
# import resnet
import nwarmup
from typing import Any, Callable, List, Optional

import torch.nn as nn
from torch import Tensor

class BN2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x

##############################################################################################################
    
def resetFHGN(modules):
    for m in modules:
        if isinstance(m, FHGN):
            m.u = []
            m.s = []
            m.time = 0

class FHGN(nn.Module):
    def __init__(self, k=2):
        super().__init__()
        self.u = []
        self.s = []
        self.time = 0
        self.k = k

    def forward(self, x):
        if self.time == 0:
            psp = 0.
            last = 0.
        else:
            psp = self.u[-1]
            last = self.s[-1]
        psp = psp * (1. - last) / 2. + x[-1]
        out = snnalgo.PiecewiseQuadratic.apply(psp - 1., 1.)
        self.u.append(psp.detach())
        self.s.append(out.detach())
        self.time += 1
        if self.k > 0 and len(self.u) > self.k:
            listu = self.u[-self.k:]
            lists = self.s[-self.k:]
            time = self.k
        else:
            listu = self.u
            lists = self.s
            time = self.time
        return FHGFunc.apply(x, torch.stack(listu, 0), torch.stack(lists, 0), time)

def surrogate_gradient(x, alpha):
    x_abs = x.abs()
    mask = (x_abs > (1 / alpha))
    return (- (alpha ** 2) * x_abs + alpha).masked_fill_(mask, 0)

class FHGFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, u, s, time):
        ctx.save_for_backward(x, u, s)
        ctx.time = time
        return s
    
    @staticmethod
    def backward(ctx, gradout):
        x, u, s = ctx.saved_tensors

        dx = torch.zeros_like(x)
        du = 0.
        for t in range(ctx.time):
            index = ctx.time - t - 1
            sg = surrogate_gradient(u[index] - 1., 1.)
            du = du / 2. * (1. - s[index] - u[index] * sg) + gradout[index] * sg
            dx[index] = du

        return dx, None, None, None

###################################################################################


def amp_train_static(net, dataloader, optimizer, scaler, args):
    net.train()
    running_loss = sumup = 0.0
    correct = 0.
    
    for i, (inputs, labels) in enumerate(dataloader):
        
        inputs = inputs.float().to(0, non_blocking=True)
        labels = labels.to(0, non_blocking=True)
        
        resetFHGN(net.modules())
        sumout = 0.
        x = []
        for t in range(args.T):
            with amp.autocast():
                if args.k > 0 and t < args.k:
                    x.append(inputs)
                out = net(torch.stack(x, 0))
                    
                loss = out.sum() * 0. + F.cross_entropy(out[-1], labels) # + 5e-2 * F.mse_loss(out, F.one_hot(labels, args.num_classes).to(out))
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
            sumout += out[-1]
    
        running_loss += loss.item()
        sumup += inputs.shape[0]
        
        correct += sumout.argmax(dim=1).eq(labels).sum().item()
        
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
        
        sumout = 0.
        resetFHGN(net.modules())
        
        for t in range(args.T):
            out = net(torch.stack([inputs], 0))
            sumout += out[-1]
            
        sumup += inputs.shape[0]
        correct += sumout.argmax(dim=1).eq(labels).sum().item()
        
    net.train()
    return correct, sumup

def execuate(device, args):
    
    tr_data, ts_data = snnvds.cifar10_dataset(args.data_dir, args.batch_size1, args.batch_size2)
    
    # neuron_cfg = {
    #     'alpha': nwarmup.PolynormialWarmup(1., 1., 200),
    #     'tau': 2.,
    #     'sg': snnalgo.PiecewiseQuadratic,
    # }
    
    # net = resnet18(OnlineLIF, {}, num_classes=10, feature=cifar10_feature).to(device)
    net = ms_resnet.resnet18('A', FHGN, {'k': args.k}, num_classes=10, feature=ms_resnet.cifar10_feature, norm_layer=BN2d).to(device)

    optimizer = torch.optim.AdamW(net.parameters(), lr=0.001 / 6., weight_decay=5e-5)
    # optimizer = torch.optim.SGD(net_bptt.parameters(), lr=0.15, weight_decay=5e-5, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    
    writer = SummaryWriter(log_dir=f'LOGS/CIFAR-10/ms_resnet18/FullHG/k-{args.k}')
    
    scaler = amp.GradScaler()
    
    max_acc = 0.
    for e in range(args.num_epochs):
        if args.verbose:
            dataloader = tqdm(tr_data)
        else:
            dataloader = tqdm(tr_data)

        correct, sumup, loss = amp_train_static(net, dataloader, optimizer, scaler, args)
        correct, sumup = validate_static(net, ts_data, device, args)
        
        max_acc = max(max_acc, correct)

        print('epoch: ', e, f'loss: {loss:.4f}, Acc: {(correct / sumup) * 100:.2f}%, Best: {(max_acc / sumup) * 100:.2f}%')
        
        writer.add_scalar('Loss', loss, e)
        writer.add_scalar('Acc', correct / sumup, e)
        
        scheduler.step()
    writer.close()
        

def main():
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size1', type=int, default=128, help='batch size for single device.')
    parser.add_argument('--batch_size2', type=int, default=256, help='test batch size.')
    
    parser.add_argument('--T', type=int, default=6, help='snn simulate time step.')
    parser.add_argument('--k', type=int, default=2)
    parser.add_argument('--num_epochs', type=int, default=200, help='max epochs for train process.')
    
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='')
    parser.add_argument('--data_dir', type=str, default='./data_dir', help='data directory.')
    
    parser.add_argument('--print_intv', type=int, default=50, 
                        help='train steps interval to print train mesasges: show messages after each {intv} batches.')

    parser.add_argument('--verbose', '-V', action='store_true', 
                        help='whether to display training progress in the master process.')
    parser.add_argument('--device', type=str, default='cuda')
    cmd_args = parser.parse_args()
    
    execuate(torch.device(cmd_args.device), cmd_args)
    

if __name__ == '__main__':
    main()