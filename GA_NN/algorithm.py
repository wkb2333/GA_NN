import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm, trange


class Child(nn.Module):
    """
    种群个体
    初始化网络结构
    """

    def __init__(self, cp_num=9, fc_num=3) -> None:
        super(Child, self).__init__()
        self.cp_num = cp_num
        self.fc_num = fc_num
        self.input_size = 28

        self.conv_args = {'out_channels': [x for x in range(3, 385, 3)],
                          'kernel_size': [x for x in range(1, 12, 2)]}
        self.pool_args = [x for x in range(15, 28)]
        self.line_args = [x for x in range(128, 4097, 128)]

        self.conv_li = []
        self.pool_li = []
        self.fc_li = []

        self.net = nn.Sequential()
        self.init_net()

    def forward(self, X):
        X = self.net(X)
        return X

    def init_net(self):
        # 进行网络结构初始化，进行必要约束
        # 考虑每一层，第一层卷积输入必然为1，最后一层线性输出必然为10,
        # 其余层均可以随机初始化并且加入遗传
        # 对于卷积层，输入通道应为上一层输出通道，而输出通道为待优化参数
        # 对于池化层，这里使用自适应池化，只需输出需要的大小
        # 卷积和池化组最后一层限定为池化层
        # 对于全连接层，输入特征应为上一层输出特征，而输出特征为待优化参数
        self.net.add_module('conv_pool', nn.Sequential())
        self.net.add_module('flat', nn.Flatten())
        self.net.add_module('linear', nn.Sequential())

        for idx in range(self.cp_num):
            self.create_cp(idx)
        for _ in range(self.fc_num):
            idx = len(self.fc_li)
            self.create_fc(idx)

    def cal_input_size(self):
        """
        计算网络从开始到此的输入图像大小
        用于确定当前层的kernel_size
        仅考虑初始化时，即自上而下添加层
        :return: int self.input_size
        """
        self.input_size = 28
        for layer in self.net.conv_pool.children():
            if type(layer) is nn.modules.conv.Conv2d:
                self.input_size = self.input_size - layer.kernel_size[0] + 1
            elif type(layer) is (nn.modules.pooling.AdaptiveMaxPool2d or nn.modules.pooling.AdaptiveAvgPool2d):
                self.input_size = layer.output_size[0]
            else:
                continue
        return self.input_size

    def adapt_attr(self, add_type, idx):
        # 需要准确定位当前层所在位置，传入index 提取出特定类型进行定位
        if add_type == nn.modules.conv.Conv2d:
            if len(self.conv_li) == 0:
                return 1
            else:
                return eval(f'self.net.conv_pool.conv{idx - 1}.out_channels')
        if add_type == (nn.modules.pooling.AdaptiveMaxPool2d or nn.modules.pooling.AdaptiveAvgPool2d):
            pass
        if add_type == nn.modules.linear.Linear:
            if len(self.fc_li) == 0:
                size = self.cal_input_size()
                layer = eval(f'self.net.conv_pool.{self.conv_li[-1]}')
                return layer.out_channels * size * size
            else:
                return eval(f'self.net.linear.linear{idx - 1}.out_features')

    def create_conv(self, idx):
        if self.cal_input_size() < 5:
            return
        out_channels = np.random.choice(self.conv_args['out_channels'])
        kernel_size = np.random.choice(range(1, self.cal_input_size(), 2))
        if idx == 0:
            self.net.conv_pool.add_module('conv' + str(idx), nn.Conv2d(1, out_channels, kernel_size))
            self.conv_li.append('conv' + str(idx))
            if np.random.rand() > 0.5:
                self.net.conv_pool.add_module('norm' + str(idx), nn.BatchNorm2d(out_channels))
            self.net.conv_pool.add_module('activate' + str(idx), nn.ReLU())
        else:
            last_out = self.adapt_attr(nn.modules.conv.Conv2d, idx)
            self.net.conv_pool.add_module('conv' + str(idx), nn.Conv2d(last_out, out_channels, kernel_size))
            self.conv_li.append('conv' + str(idx))
            if np.random.rand() > 0.5:
                self.net.conv_pool.add_module('norm' + str(idx), nn.BatchNorm2d(out_channels))
            self.net.conv_pool.add_module('activate' + str(idx), nn.ReLU())

    def create_pool(self, idx):
        # 这里限制最小池化尺寸
        if self.cal_input_size() < 5:
            return
        output_size = np.random.choice(range(3, self.cal_input_size()))
        if np.random.rand() > 0.5:
            self.net.conv_pool.add_module('pool' + str(idx), nn.AdaptiveMaxPool2d((output_size, output_size)))
            self.pool_li.append('pool' + str(idx))
        else:
            self.net.conv_pool.add_module('pool' + str(idx), nn.AdaptiveAvgPool2d((output_size, output_size)))
            self.pool_li.append('pool' + str(idx))

    def create_fc(self, idx):
        last_out = self.adapt_attr(nn.modules.linear.Linear, idx)
        out_features = np.random.choice(self.line_args)
        # if idx == 0:
        #     in_features = self.net.conv_pool[-1].output_size**2
        #     self.net.linear.add_module('linear' + str(idx), nn.Linear(in_features, out_features))
        #     self.fc_li.append('linear' + str(idx))
        #     if np.random.rand() > 0.5:
        #         self.net.linear.add_module('norm' + str(idx), nn.BatchNorm2d(1))
        #     self.net.linear.add_module('activate' + str(idx), nn.ReLU())
        if idx == self.fc_num - 1:
            self.net.linear.add_module('linear' + str(idx), nn.Linear(last_out, 10))
            self.fc_li.append('linear' + str(idx))
        else:
            self.net.linear.add_module('linear' + str(idx), nn.Linear(last_out, out_features))
            self.fc_li.append('linear' + str(idx))
            if np.random.rand() > 0.5:
                self.net.linear.add_module('norm' + str(idx), nn.BatchNorm1d(out_features))
            self.net.linear.add_module('activate' + str(idx), nn.ReLU())

    def create_cp(self, idx):
        if idx == 0:
            module_num = len(self.conv_li)
            self.create_conv(module_num)
        elif idx == self.cp_num - 1:
            module_num = len(self.pool_li)
            self.create_pool(module_num)
        elif np.random.rand() > 0.5:
            module_num = len(self.conv_li)
            self.create_conv(module_num)
        else:
            module_num = len(self.pool_li)
            self.create_pool(module_num)

    def mutation(self):
        pass
        # 这里变异只进行修改，不进行增删
        # length = self.cp_num + self.fc_num
        # if np.random.randint(0, length) < self.cp_num:
        #     mutate = np.random.choice(self.conv_li + self.pool_li[0:-1])  # 这里是为了避免选择到最后一层导致线性层发生变化
        #     module_type = mutate[0:-1]
        #     idx = int(mutate[-1])
        #     if module_type == 'conv':
        #         out_channels = np.random.choice(self.conv_args['out_channels'])
        #         kernel_size = np.random.choice(self.conv_args['kernel_size'])
        #         eval(f"self.net.conv_pool.{mutate}"
        #              f" = nn.Conv2d(self.net.conv_pool.{mutate}.in_channels, out_channels, kernel_size)")
        #         eval(f"self.net.conv_pool.{module_type + str(idx+1)} = "
        #              f"nn.Conv2d(self.net.conv_pool.{mutate}.out_channels, "
        #              f"self.net.conv_pool.{module_type + str(idx+1)}.out_channels, "
        #              f"self.net.conv_pool.{module_type + str(idx+1)}.kernel_size)")
        #     elif module_type == 'pool':
        #         # 此处变异只使用MaxPooling
        #         eval(f"self.net.conv_pool.{mutate}"
        #              f" = nn.AdaptiveMaxPool2d(np.random.choice(self.pool_args))")
        # else:
        #     mutate = np.random.choice(self.fc_li[0:-1])
        #     idx = int(mutate[-1])
        #     eval(f"self.net.linear.{mutate} "
        #          f"= nn.Linear(self.net.linear.{mutate}.in_features, np.random.choice(self.line_args))")
        #     eval(f"self.net.linear.{mutate[0:-1] + str(idx+1)} "
        #          f"= nn.Linear(self.net.linear.{mutate}.out_features, "
        #          f"self.net.linear.{mutate[0:-1] + str(idx+1)}.out_features)")

    def fitness_eval(self, train_loader, test_loader):
        # 训练模型
        for epoch in trange(10):
            for batch_idx, (data, target) in enumerate(train_loader):
                # 将数据转换为模型需要的格式
                # data, target = data.cuda(), target.cuda()
                # 前向传播
                output = self.net(data)
                # 计算损失
                criterion = nn.CrossEntropyLoss()
                loss = criterion(output, target)
                # 反向传播
                optimizer = optim.Adam(self.net.parameters())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        # 评估模型
        correct = 0
        total = 0
        for data, target in test_loader:
            # 将数据转换为模型需要的格式
            # data, target = data.cuda(), target.cuda()
            # 前向传播
            output = self.net(data)
            # 获取预测结果
            predict = output.argmax()
            # 计算准确率
            correct += predict.eq(target.view_as(predict)).sum().item()
            total += target.size(0)
        accuracy = correct / total
        # 返回准确率
        return accuracy


class GA:
    """
    遗传算法，针对使用pytorch进行CNN网络结构探索，需要实现：
    种群初始化
    适应度评估--
    BTS(Binary Tournament Selection)
    子代生成--
    交叉变异
    自然选择
    在得到最终的网络结构后完整训练
    """

    def __init__(self) -> None:
        self.population = self.init_population()

    def init_population(self, cp_num=9, fc_num=3, population_size=100):
        population = []
        for _ in range(population_size):
            child = Child(cp_num, fc_num)
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
