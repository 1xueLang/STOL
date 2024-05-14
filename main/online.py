import math
import torch
import torch.nn as nn
import snetx.snn.algorithm as snnalgo

THRESHOLD = 1.

# def surrogate_gradient(x, alpha):
#     return alpha / 2 / (1 + (math.pi / 2 * alpha * x).pow_(2))


def surrogate_gradient(x, alpha):
    x_abs = x.abs()
    mask = (x_abs > (1 / alpha))
    return (- (alpha ** 2) * x_abs + alpha).masked_fill_(mask, 0)


class NoneResetOnLine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, xlast, ulast, slast, tau, alpha=1., mode='s'):
        ctx.tau = tau
        ctx.alpha = alpha
        ctx.mode = mode
        ctx.save_for_backward(u, ulast)
        ctx.no_gradx = xlast is None
        
        s = u.ge(THRESHOLD).to(u)
        
        if xlast is None:
            pass
        else:
            s = torch.stack([s, slast], dim=0)
        
        return s
    
    @staticmethod
    def backward(ctx, grad_out):
        u, ulast = ctx.saved_tensors
        
        grad1, grad2 = grad_out
        grad1 = grad1 * surrogate_gradient(u - THRESHOLD, ctx.alpha)
        
        if ctx.no_gradx:
            grad2 = None
        else:
            if ctx.mode == 's':
                grad2 = grad2 * surrogate_gradient(ulast - THRESHOLD, ctx.alpha) + grad1 / ctx.tau
            elif ctx.mode == 't':
                grad2 = grad1 / ctx.tau
            else:
                grad2 = torch.zeros_like(grad2)
            
        return grad1, grad2, None, None, None, None, None

class HardResetOnLine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, xlast, ulast, slast, tau, alpha=1., mode='s'):
        ctx.tau = tau
        ctx.alpha = alpha
        ctx.mode = mode
        ctx.no_gradx = xlast is None
        ctx.save_for_backward(u, ulast, slast)
        
        s = u.ge(THRESHOLD).to(u)
        
        if xlast is None:
            pass
        else:
            s = torch.stack([s, slast], dim=0)
        
        return s
    
    @staticmethod
    def backward(ctx, grad_out):
        
        u, ulast, slast = ctx.saved_tensors
            
        grad1, grad2 = grad_out
        grad1 = grad1 * surrogate_gradient(u - THRESHOLD, ctx.alpha)
        
        if ctx.no_gradx:
            grad2 = None
        else:
            sg = surrogate_gradient(ulast - THRESHOLD, ctx.alpha)
            temporal_grad = grad1 / ctx.tau * (1. - slast - sg * ulast)
            if ctx.mode == 's':
                grad2 = grad2 * sg + temporal_grad
            elif ctx.mode == 't':
                grad2 = temporal_grad
            else:
                grad2 = torch.zeros_like(grad2)
            
        return grad1, grad2, None, None, None, None, None


class neuronOnLine(nn.Module):
    def __init__(self, tau=10., alpha=lambda : 1., reset='hard', mode='s'):
        super().__init__()
        self.tau = tau
        self.alpha = alpha
        self.ulast = None
        self.slast = None
        self.reset = reset
        self.mode = mode
        if reset == 'hard':
            self.spiking = HardResetOnLine.apply
        elif reset == 'none':
            self.spiking = NoneResetOnLine.apply
        else:
            raise NotImplementedError(f'{reset} not supported.')
        
    def forward(self, inputs):
        if self.training:
            x, xlast = inputs
        else:
            x, xlast = inputs, None
        
        self.reset_batch(x)
        
        u = self.neuron_charge(x)
        s = self.spiking(u, xlast, self.ulast, self.slast, self.tau, self.alpha(), self.mode)
            
        self._post_firing(u, s[0] if self.training else s)
        
        return s
    
    def neuron_charge(self, x):
        if self.reset == 'hard':
            u = self.ulast * (1. - self.slast)
        else:
            u = self.ulast
        return u / self.tau + x
    
    def reset_batch(self, x):
        if self.slast == None or self.slast.shape != x.shape:
            self.ulast = self.slast = torch.zeros_like(x)
    
    def reset_period(self):
        self.ulast = None
        self.slast = None

    def _post_firing(self, u, s):
        self.ulast = u.detach()
        self.slast = s.detach()
    

def double_forwrad(nn_block, inputs, training):
    if not training:
        out = nn_block(inputs)
    else:
        x, histx = inputs
        h = nn_block(x)
        histh = nn_block(histx)
        out = (h, histh)
    return out

class DoubleForward(nn.Module):
    def __init__(self, nn_block):
        super().__init__()
        self.nn_block = nn_block
    
    def forward(self, inputs):
        return double_forwrad(self.nn_block, inputs, self.training)
    
def single_forward(nn_block, inputs, training):
    if training:
        return nn_block(inputs[0])
    else:
        return nn_block(inputs)

class SingleForward(nn.Module):
    def __init__(self, nn_block):
        super().__init__()
        self.nn_block = nn_block
    
    def forward(self, inputs):
        return single_forward(self.nn_block, inputs, self.training)

class Flattten(object):
    def __init__(self, dim=1):
        self.dim=1
        
    def __call__(self, inputs):
        return torch.flatten(inputs, self.dim)
        
def neuronOnLine_reset(modules):
    for m in modules:
        if isinstance(m, neuronOnLine):
            m.reset_period()
            
class Dropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.last = None
        
    def forward(self, inputs):
        if self.training:
            x, xhist = inputs
            mask = torch.rand_like(x).le(self.p).to(x)
            x = x * (1. - mask)
            if self.last != None and self.last.shape == xhist.shape:
                xhist = xhist * (1. - self.last)
            self.last = mask
            return (x, xhist)
        else:
            return nn.functional.dropout(inputs, self.p, self.training)
        
    def __repr__(self):
        return f'online.Dropout(p={self.p})'