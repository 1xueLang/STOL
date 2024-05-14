import torch
import nwarmup
import resnet
import online
from snetx.models import ms_resnet
from snetx.snn import neuron


class BN(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(in_channels)
        
    def forward(self, x):
        out = []
        for i in range(x.shape[1]):
            out.append(self.bn(x[:, i]))
        return torch.stack(out, dim=1)
    
def load_pretrained(net, ckpt):
    net_state = net.state_dict()
    net_state['feature.0.nn_block.weight'] = ckpt['conv1.weight']
    net_state['feature.1.nn_block.weight'] = ckpt['bn1.weight']
    net_state['feature.1.nn_block.bias'] = ckpt['bn1.bias']
    net_state['feature.1.nn_block.running_mean'] = ckpt['bn1.running_mean']
    net_state['feature.1.nn_block.running_var'] = ckpt['bn1.running_var']
    net_state['feature.1.nn_block.num_batches_tracked'] = ckpt['bn1.num_batches_tracked']
    net_state['fc.1.nn_block.bias'] = ckpt['fc.bias']
    net_state['fc.1.nn_block.weight'] = ckpt['fc.weight']
    
    for k in ckpt.keys():
        if 'layer' in k:
            if 'downsample' not in k:
                _k = ''.join([k[:len('layer1.2.conv1.')], k[len('layer1.2.conv1.module.')], '.', 'nn_block', k[len('layer1.2.conv1.module.0'):]])
            else:
                _k = ''.join([k[:len('layer2.0.downsample.0.')], k[len('layer2.0.downsample.0.module.')], '.nn_block', k[len('layer2.0.downsample.0.module.0'):]])
            print(f'Load: [{k}]')
            print(f'====> [{_k}]')
            net_state[_k] = ckpt[k]
            
    net.load_state_dict(net_state)

        
def load_from_bptt(net, ckpt):
    net_state = net.state_dict()
    net_state['feature.0.nn_block.weight'] = ckpt['feature.0.model.weight']
    net_state['feature.1.nn_block.weight'] = ckpt['feature.1.bn.weight']
    net_state['feature.1.nn_block.bias'] = ckpt['feature.1.bn.bias']
    net_state['feature.1.nn_block.running_mean'] = ckpt['feature.1.bn.running_mean']
    net_state['feature.1.nn_block.running_var'] = ckpt['feature.1.bn.running_var']
    net_state['feature.1.nn_block.num_batches_tracked'] = ckpt['feature.1.bn.num_batches_tracked']
    net_state['fc.1.nn_block.bias'] = ckpt['fc.bias']
    net_state['fc.1.nn_block.weight'] = ckpt['fc.weight']
    
    c = 8
    for k in ckpt.keys():
        if 'layer' in k:
            if 'downsample' not in k:
                if 'model' in k:
                    _k = ''.join([k[:len('layer1.0.conv1.0.')], 'nn_block.weight'])
                else:
                    _k = ''.join([k[:len('layer1.0.conv1.0.')], 'nn_block.', k[len('layer1.1.conv2.1.bn.'):]])
            else:
                if 'model' in k:
                    _k = ''.join([k[:len('layer4.0.downsample.1.0.')], 'nn_block.weight'])
                else:
                    _k = ''.join([k[:len('layer4.0.downsample.1.1.')], 'nn_block.', k[len('layer4.0.downsample.1.1.bn.'):]])
            # print(f'Load: [{k}]')
            # print(f'====> [{_k}]')
            net_state[_k] = ckpt[k]
            c += 1
            
    net.load_state_dict(net_state)
    # print('count', c)
# for n, p in net_bptt.named_parameters():
#     print(n)
    
# for n, p in net_stol_s.named_parameters():
#     print(n)

# for p1, p2 in zip(net_bptt.named_parameters(), net_stol_s.named_parameters()):
    # if 'weight' in p1[0] and 'bn' not in p1[0] and 'downsample' not in p1[0]: 
