import torch
from torch.cuda import amp
import torch.nn.functional as F
import online

def amp_train_dvs(net, dataloader, optimizer, scaler, args):
    net.train()
    running_loss = sumup = 0.0
    correct = 0.
    
    for i, (inputs, labels) in enumerate(dataloader):
        
        inputs = inputs.float().to(0, non_blocking=True)
        labels = labels.to(0, non_blocking=True)
        
        online.neuronOnLine_reset(net.modules())
        histx = 0.
        sumout = 0.
            
        for t in range(args.T):
            if t == 0:
                histx = torch.zeros_like(inputs[:, 0])
            with amp.autocast():
                out = net((inputs[:, t], histx))[0]
                loss = F.cross_entropy(out, labels) + 5e-2 * F.mse_loss(out, F.one_hot(labels, args.num_classes).to(out))
                
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
            histx = inputs[:, t]
            sumout += out
    
        running_loss += loss.item()
        sumup += inputs.shape[0]
        
        correct += sumout.argmax(dim=1).eq(labels).sum().item()
        
        if (1 + i) % args.print_intv == 0:
            print(f'  Loss: {loss.item():.4f}, Acc: {(correct / sumup) * 100:.2f}%')
        
    return correct, sumup, float(running_loss / len(dataloader))

def train_dvs(net, dataloader, optimizer, args):
    net.train()
    running_loss = sumup = 0.0
    correct = 0.
    
    for i, (inputs, labels) in enumerate(dataloader):
        
        inputs = inputs.float().to(0, non_blocking=True)
        labels = labels.to(0, non_blocking=True)
        
        online.neuronOnLine_reset(net.modules())
        histx = 0.
        sumout = 0.
            
        for t in range(args.T):
            if t == 0:
                histx = torch.zeros_like(inputs[:, 0])
            out = net((inputs[:, t], histx))[0]
            loss = F.cross_entropy(out, labels) + 5e-2 * F.mse_loss(out, F.one_hot(labels, args.num_classes).to(out))
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                
            histx = inputs[:, t]
            sumout += out
    
        running_loss += loss.item()
        sumup += inputs.shape[0]
        
        correct += sumout.argmax(dim=1).eq(labels).sum().item()
        
        if (1 + i) % args.print_intv == 0:
            print(f'  Loss: {loss.item():.4f}, Acc: {(correct / sumup) * 100:.2f}%')
        
    return correct, sumup, float(running_loss / len(dataloader))


def amp_train_static(net, dataloader, optimizer, scaler, args):
    net.train()
    running_loss = sumup = 0.0
    correct = 0.
    
    for i, (inputs, labels) in enumerate(dataloader):
        
        inputs = inputs.float().to(0, non_blocking=True)
        labels = labels.to(0, non_blocking=True)
        
        online.neuronOnLine_reset(net.modules())
        histx = 0.
        sumout = 0.
            
        for t in range(args.T):
            if t == 0:
                histx = torch.zeros_like(inputs)
            with amp.autocast():
                out = net((inputs, histx))[0]
                loss = F.cross_entropy(out, labels) # + 5e-2 * F.mse_loss(out, F.one_hot(labels, args.num_classes).to(out))
                
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
            histx = inputs
            sumout += out
    
        running_loss += loss.item()
        sumup += inputs.shape[0]
        
        correct += sumout.argmax(dim=1).eq(labels).sum().item()
        
        if (1 + i) % args.print_intv == 0:
            print(f'  Loss: {loss.item():.4f}, Acc: {(correct / sumup) * 100:.2f}%')
        
    return correct, sumup, float(running_loss / len(dataloader))

def train_static(net, dataloader, optimizer, args):
    net.train()
    running_loss = sumup = 0.0
    correct = 0.
    
    for i, (inputs, labels) in enumerate(dataloader):
        
        inputs = inputs.float().to(0, non_blocking=True)
        labels = labels.to(0, non_blocking=True)
        
        online.neuronOnLine_reset(net.modules())
        histx = 0.
        sumout = 0.
            
        for t in range(args.T):
            if t == 0:
                histx = torch.zeros_like(inputs)
            out = net((inputs, histx))[0]
            loss = F.cross_entropy(out, labels) #+ 5e-2 * F.mse_loss(out, F.one_hot(labels, args.num_classes).to(out))
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                
            histx = inputs
            sumout += out
    
        running_loss += loss.item()
        sumup += inputs.shape[0]
        
        correct += sumout.argmax(dim=1).eq(labels).sum().item()
        
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

        sumout = 0.        
        online.neuronOnLine_reset(net.modules())
        histx = 0.
        for t in range(args.T):
            x = inputs[:, t]
            out = net(x)
            sumout += out
            histx = x
            
        sumup += inputs.shape[0]
        correct += sumout.argmax(dim=1).eq(labels).sum().item()
        
    net.train()
    return correct, sumup

@torch.no_grad()
def validate_static(net, dataloader, device, args,):
    net.eval()
    correct = 0.
    sumup = 0.
    for inputs, labels in dataloader:
        inputs = inputs.float().to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        sumout = 0.
        online.neuronOnLine_reset(net.modules())
        
        for t in range(args.T):
            out = net(inputs)
            sumout += out
            
        sumup += inputs.shape[0]
        correct += sumout.argmax(dim=1).eq(labels).sum().item()
        
    net.train()
    return correct, sumup
