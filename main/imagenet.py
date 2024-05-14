import os
import argparse
from tqdm import tqdm
from datetime import datetime

import torch
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

import snetx
from snetx import utils
import snetx.snn.algorithm as snnalgo

import datasets.vision as visionds

import nwarmup
import resnet

import online
import load

class FineTune(object):
    def __init__(self, T):
        self.T = T

    def __call__(self, epoch):
        if epoch == 0:
            return 0.01
        else:
            return (1e-3 - (1e-3 - 1e-5) * epoch / self.T)

def train(net, dataloader, optimizer, scaler, args):
    net.train()
    running_loss = sumup = 0.0
    correct = 0.
    top5 = 0.
    
    for i, (inputs, labels) in enumerate(dataloader):
        
        inputs = inputs.float().to(0, non_blocking=True)
        labels = labels.to(0, non_blocking=True)
        
        online.neuronOnLine_reset(net.modules())
        histx = 0.
        sumout = 0.
            
        for t in range(args.T):
            if t == 0:
                histx = torch.zeros_like(inputs)
            
            if scaler is None:
                out = net((inputs, histx))[0]
                loss = F.cross_entropy(out, labels)# + 5e-2 * F.mse_loss(out, F.one_hot(labels, args.num_classes).to(out))
                    
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                with amp.autocast():
                    out = net((inputs, histx))[0]
                    loss = F.cross_entropy(out, labels)# + 5e-2 * F.mse_loss(out, F.one_hot(labels, args.num_classes).to(out))
                
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                
            histx = inputs
            sumout += out
    
        running_loss += loss.item()
        sumup += inputs.shape[0]
        out = out.sum(dim=1)
        
        top5 += (sumout.topk(5, dim=1, largest=True, sorted=True)[1]).eq(labels.unsqueeze(dim=1)).sum().item()
        correct += sumout.argmax(dim=1).eq(labels).sum().item()
        
        if (1 + i) % args.print_intv == 0:
            print(f'  Loss: {loss.item():.4f}, Acc: {(correct / sumup) * 100:.2f}%, Top5: {(top5 / sumup) * 100:.2f}')
        
    return correct, top5, sumup, float(running_loss / len(dataloader))


@torch.no_grad()
def validate(net, dataloader, device, args,):
    net.eval()
    correct = 0.
    top5 = 0.
    sumup = 0.
    for inputs, labels in tqdm(dataloader):
        inputs = inputs.float().to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        sumout = 0.        
        online.neuronOnLine_reset(net.modules())
        histx = 0.
        for t in range(args.T):
            x = inputs
            out = net(x)
            sumout += out
            histx = x
            
        sumup += inputs.shape[0]
        correct += sumout.argmax(dim=1).eq(labels).sum().item()
        top5 += (sumout.topk(5, dim=1, largest=True, sorted=True)[1]).eq(labels.unsqueeze(dim=1)).sum().item()
        
    net.train()
    return correct, top5, sumup


def execuate(device, args):
    
    if args.debug:
        writer = None
    else:
        writer = SummaryWriter(log_dir=f'{args.logs_dir}/{args.dataset}/{args.arch}/{args.T}')
    
    tr_data, ts_data = visionds.imagenet_dataset(args.data_dir, args.batch_size1, args.batch_size2)
    
    neuron_cfg = {
        # 'alpha': nwarmup.ConsineAnnealingWarmup(args.base, args.bound, args.T_max2), 
        'alpha': nwarmup.PolynormialWarmup(args.base, args.bound, args.T_max2),
        'tau': args.tau,
        'mode': args.mode,
        'reset': args.reset,
    }
    if 'sew' in args.arch:
        net = resnet.sew_resnet.__dict__[args.arch[3:]](online.neuronOnLine, neuron_cfg, num_classes=args.num_classes).to(device)
        if args.pretrained:
            load.load_pretrained(net, torch.load(args.pretrained_weight_path))
        print(net)
        net.train()
    elif 'ms' in args.arch:
        net = resnet.ms_resnet.__dict__[args.arch[2:]]('A', online.neuronOnLine, neuron_cfg, num_classes=args.num_classes).to(device)
        if args.pretrained:
            load.load_pretrained(net, torch.load(args.pretrained_weight_path))
        print(net)
        net.train()
    else:
        raise ValueError(f'{args.arch} not supported.')
    
    if args.amp:
        scaler = amp.GradScaler()
    else:
        scaler = None
        
    # optimizer = torch.optim.AdamW(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    optimizer = torch.optim.SGD(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max1)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, FineTune(args.T_max1))
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
    
    # correct, top5, sumup = validate(net, ts_data, device, args)
    # print(f'Test PreTrained: top1-{correct / sumup}, top5-{top5 / sumup}')
    max_top1 = max_top5 = 0.
    for e in range(args.num_epochs):
        if args.verbose:
            dataloader = tqdm(tr_data)
        else:
            dataloader = tqdm(tr_data)
        
        correct, top5, sumup, loss = train(net, dataloader, optimizer, scaler, args)
        correct, top5, sumup = validate(net, ts_data, device, args)
          
        if not args.debug:      
            writer.add_scalar('Loss', loss, e)
            writer.add_scalar('Acc', correct / sumup, e)
        
        max_top1 = max(max_top1, correct)
        max_top5 = max(max_top5, top5)

        print('epoch: ', e, f'loss: {loss:.4f}, Acc: {(correct / sumup) * 100:.2f}%, Top5: {(top5 / sumup) * 100:.2f}, Best: {(max_top1 / sumup) * 100:.2f}%-{(max_top5 / sumup) * 100:.2f}%')  
        print(scheduler.get_last_lr(), neuron_cfg['alpha'].get_last_alpha())
        
        scheduler.step()
        neuron_cfg['alpha'].step()
    
    writer.close()

def main():
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size1', type=int, default=200, help='batch size for single device.')
    parser.add_argument('--batch_size2', type=int, default=128, help='test batch size.')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='learning rate for gradient descent.')
    parser.add_argument('--weight_decay', type=float, default=0., help='penal term parameter for model weight.')
    
    parser.add_argument('--tau', type=float, default=2., help='')
    parser.add_argument('--reset', type=str, default='hard', help='')
    parser.add_argument('--T', type=int, default=6, help='snn simulate time step.')
    parser.add_argument('--num_epochs', type=int, default=100, help='max epochs for train process.')
    parser.add_argument('--T_max1', type=int, default=100, help='schedule period for consine annealing lr scheduler.')
    parser.add_argument('--T_max2', type=int, default=100, help='schedule period for consine annealing neuron warmup.')
    parser.add_argument('--base', type=float, default=1., help='')
    parser.add_argument('--bound', type=float, default=1., help='')
    parser.add_argument('--drop', type=float, default=0.2, help='')
    parser.add_argument('--mode', type=str, default='s', help='')
    
    parser.add_argument('--dataset', type=str, default='ImageNet', help='')
    parser.add_argument('--data_dir', type=str, default='./data_dir/ImageNet2012', help='data directory.')
    parser.add_argument('--logs_dir', type=str, default='LOGS/logs', help='logs directory.')
    
    parser.add_argument('--print_intv', type=int, default=50, 
                        help='train steps interval to print train mesasges: show messages after each {intv} batches.')
    parser.add_argument('--amp', '-A', action='store_true')

    parser.add_argument('--verbose', '-V', action='store_true', 
                        help='whether to display training progress in the master process.')
    parser.add_argument('--arch', type=str, default='sewresnet34', 
                        help='network architecture.'
                             'sewewsnet{}: sew resnet 18, 34, 50, 101'
                             'msresnet{}: ms resnet 18, 34, 50, 101'
                        )
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--debug', '-D', action='store_true')
    parser.add_argument('--pretrained', '-P', action='store_true')
    parser.add_argument('--pretrained_weight_path', type=str, default='./pretrained/sew34.pth')
    
    cmd_args = parser.parse_args()
    
    execuate(torch.device(cmd_args.device), cmd_args)
    

if __name__ == '__main__':
    main()