import torch
import torch.nn as nn
import numpy as np

class Child(nn.Module):
    def __init__(self) -> None:
        super.__init__()
        self.cp_num = 8
        self.fc_num = 3

        self.module = nn.Sequential()
        self.module.add_module('cp', nn.Sequential())
        self.module.add_module('linear', nn.Sequential())

        self.conv_args = {'in_channels': 1, 'out_channels': 3, 'kernel_size': 3}
        self.pool_args = {'output_size': (15, 15)}
        self.line_args = {'in_features': 225, 'out_features': 10}

        self.module.cp.add_module(nn.Conv2d(1, 3, 3))
        for _ in range(self.cp_num-1):
            self.create_cp()
        for _ in range(self.fc_num):
            self.create_fc()

    def create_conv(self):
        self.module.cp.add_module('conv', nn.Conv2d(**self.conv_args))
        self.module.cp.add_module(nn.ReLU())

    def create_pool(self):
        if np.random.rand() > 0.5:
            self.module.linear.add_module('pool', nn.AdaptiveMaxPool2d(**self.pool_args))
        else:
            self.module.linear.add_module('pool', nn.AdaptiveAvgPool2d(**self.pool_args))
            
    def create_fc(self):
        self.module.add_module('linear', nn.Linear(**self.line_args))
        self.module.add_module(nn.ReLU())

    def create_cp(self):
        if np.random.rand() > 0.5:
            self.create_conv()
        else:
            self.create_pool()

    def mutation():
        pass