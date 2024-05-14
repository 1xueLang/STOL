from typing import Any, Callable, List, Optional, Dict, TypeVar

import torch
import torch.nn as nn
from . import sew_resnet
from . import ms_resnet

import online

class ResNet19(nn.Module):
    def __init__(
        self,
        type,
        drop,
        neuron,
        neuron_cfg,
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        feature: Callable = None
        ):
        
        super().__init__()
        hidden = 512
        if type[:2] == 'ms':
            self.resnet = ms_resnet.resnet18(
                'A',
                neuron,
                neuron_cfg,
                num_classes=hidden,
                zero_init_residual=zero_init_residual,
                groups=groups,
                width_per_group=width_per_group,
                replace_stride_with_dilation=replace_stride_with_dilation,
                norm_layer=norm_layer,
                feature=feature
            )
        elif type[:3] == 'sew':
            self.resnet = sew_resnet.resnet18(
                neuron,
                neuron_cfg,
                num_classes=hidden,
                zero_init_residual=zero_init_residual,
                groups=groups,
                width_per_group=width_per_group,
                replace_stride_with_dilation=replace_stride_with_dilation,
                norm_layer=norm_layer,
                feature=feature
            )
        else:
            pass
        
        self.out_layer = nn.Sequential(
            neuron(**neuron_cfg),
            online.Dropout(p=drop),
            online.DoubleForward(nn.Linear(hidden, num_classes))
        )
    
    def forward(self, x):
        return self.out_layer(self.resnet(x))


class ResNet20(nn.Module):
    def __init__(
        self,
        type,
        drop,
        neuron,
        neuron_cfg,
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        feature: Callable = None
        ):
        super().__init__()
        hidden = 512
        if type[:2] == 'ms':
            self.resnet = ms_resnet.resnet18(
                'A',
                neuron,
                neuron_cfg,
                num_classes=hidden,
                zero_init_residual=zero_init_residual,
                groups=groups,
                width_per_group=width_per_group,
                replace_stride_with_dilation=replace_stride_with_dilation,
                norm_layer=norm_layer,
                feature=feature
            )
        elif type[:3] == 'sew':
            self.resnet = sew_resnet.resnet18(
                neuron,
                neuron_cfg,
                num_classes=hidden,
                zero_init_residual=zero_init_residual,
                groups=groups,
                width_per_group=width_per_group,
                replace_stride_with_dilation=replace_stride_with_dilation,
                norm_layer=norm_layer,
                feature=feature
            )
        else:
            pass
        
        self.out_layer = nn.Sequential(
            online.Dropout(p=drop),
            neuron(**neuron_cfg),
            online.DoubleForward(nn.Linear(hidden, 256)),
            online.Dropout(p=drop),
            neuron(**neuron_cfg),
            online.DoubleForward(nn.Linear(256, num_classes))
        )
    
    def forward(self, x):
        return self.out_layer(self.resnet(x))