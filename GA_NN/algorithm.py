import numpy as np
import torch
import torch.nn as nn


class Child(nn.Module):
    """
    种群个体
    初始化网络结构
    """
    def __init__(self, cp_num=9, fc_num=3) -> None:
        super(Child, self).__init__()
        self.cp_num = cp_num
        self.fc_num = fc_num

        self.conv_args = {'out_channels': [x for x in range(3, 385, 3)],
                          'kernel_size': [x for x in range(1, 16, 2)]}
        self.pool_args = [x for x in range(15, 28)]
        self.line_args = [x for x in range(128, 4097, 128)]

        self.net = nn.Sequential()
        self.init_net()

    def init_net(self):
        # 进行网络结构初始化，进行必要约束
        # 考虑每一层，第一层卷积输入必然为1，最后一层线性输出必然为10,
        # 其余层均可以随机初始化并且加入遗传
        # 对于卷积层，输入通道应为上一层输出通道，而输出通道为待优化参数
        # 对于池化层，这里使用自适应池化，只需输出需要的大小
        # 对于全连接层，输入特征应为上一层输出特征，而输出特征为待优化参数
        self.net.add_module('conv_pool', nn.Sequential())
        self.net.add_module('flat', nn.Flatten())
        self.net.add_module('linear', nn.Sequential())

        for idx in range(self.cp_num):
            self.create_cp(str(idx))
        for idx in range(self.fc_num):
            self.create_fc(str(idx))

    def adapt_attr(self, add_type):
        # 需要准确定位当前层所在位置，传入index 提取出特定类型进行定位
        if add_type == nn.modules.conv.Conv2d:
            for conv in list(self.net.conv_pool.children())[::-1]:
                if conv == add_type:
                    return conv.out_channels
        if add_type == nn.modules.pooling.AdaptiveMaxPool2d or nn.modules.pooling.AdaptiveAvgPool2d:
            pass
        if add_type == nn.modules.linear.Linear:
            for line in list(self.net.linear.children())[::-1]:
                if line == add_type:
                    return line.out_channels

    def create_conv(self, idx):
        out_channels = np.random.choice(self.conv_args['out_channels'])
        kernel_size = np.random.choice(self.conv_args['kernel_size'])
        if idx == '0':
            self.net.conv_pool.add_module('conv' + idx, nn.Conv2d(1, out_channels, kernel_size))
            if np.random.rand() > 0.5:
                self.net.conv_pool.add_module('norm'+idx, nn.BatchNorm2d(out_channels))
            self.net.conv_pool.add_module('activate' + idx, nn.ReLU())
        else:
            last_out = self.adapt_attr(nn.modules.conv.Conv2d)
            self.net.conv_pool.add_module('conv'+idx, nn.Conv2d(last_out, out_channels, kernel_size))
            if np.random.rand() > 0.5:
                self.net.conv_pool.add_module('norm'+idx, nn.BatchNorm2d(out_channels))
            self.net.conv_pool.add_module('activate'+idx, nn.ReLU())

    def create_pool(self, idx):
        output_size = np.random.choice(self.pool_args)
        if np.random.rand() > 0.5:
            self.net.conv_pool.add_module('pool'+idx, nn.AdaptiveMaxPool2d((output_size, output_size)))
        else:
            self.net.conv_pool.add_module('pool'+idx, nn.AdaptiveAvgPool2d((output_size, output_size)))
            
    def create_fc(self, idx):
        self.net.linear.add_module('linear'+idx, nn.Linear(**self.line_args))
        if np.random.rand() > 0.5:
            self.net.conv_pool.add_module('norm'+idx, nn.BatchNorm2d())
        self.net.linear.add_module('activate'+idx, nn.ReLU())

    def create_cp(self, idx):
        if np.random.rand() > 0.5:
            self.create_conv(idx)
        else:
            self.create_pool(idx)

    def mutation(self):
        pass


class GA:
    """
    遗传算法，针对使用pytorch进行CNN网络结构探索，需要实现：
    种群初始化
    适应度评估
    BTS(Binary Tournament Selection)
    子代生成
    交叉变异
    自然选择
    在得到最终的网络结构后完整训练
    """
    def __init__(self) -> None:
        self.population = self.init_population()

    def init_population(self, cp_num=9, fc_num=3, population_size=100):
        population = []
        for _ in range(population_size):
            child = Child()
            population.append(child)

        return population

    def fitness_eval(self):
        pass

    def binary_tournament_select(self):
        pass

    def offspring_generate(self):
        pass

    def crossover(self):
        pass

    def mutation(self):
        pass

    def environment_select(self):
        pass


