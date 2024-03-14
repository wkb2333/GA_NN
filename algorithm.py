import numpy as np
import torch.cuda
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

        self.conv_args = {'out_channels': [x for x in range(3, 64, 3)],
                          'kernel_size': [x for x in range(1, 8, 2)]}
        self.pool_args = [x for x in range(15, 28)]
        self.line_args = [x for x in range(128, 4097, 128)]

        self.conv_li = []
        self.pool_li = []
        self.fc_li = []

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
        self.net.to(self.device)

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
            elif type(layer) is torch.nn.modules.pooling.AdaptiveMaxPool2d:
                self.input_size = layer.output_size[0]
            elif type(layer) is torch.nn.modules.pooling.AdaptiveAvgPool2d:
                self.input_size = layer.output_size[0]
            else:
                continue
        return self.input_size

    def net_format(self):
        for idx in range(len(self.conv_li)-1):
            exec(f'self.net.conv_pool.conv{idx+1}.in_channels = self.net.conv_pool.conv{idx}.out_channels')
            for layer in self.net.conv_pool.children():
                if type(layer) is nn.modules.batchnorm.BatchNorm2d:
                    exec(f'self.net.conv_pool.norm{idx}.in_channels = nn.BatchNorm2d(self.net.conv_pool.conv{idx}.out_channels)')
        for idx in range(1, len(self.fc_li)):
            exec(f'self.net.linear.linear{idx}.in_features = self.net.linear.linear{idx-1}.out_features')
        size = self.cal_input_size()
        layer = eval(f'self.net.conv_pool.{self.conv_li[-1]}')
        self.net.linear.linear0.in_features = layer.out_channels * size * size

    def adapt_attr(self, add_type, idx):
        # 需要准确定位当前层所在位置，传入index 提取出特定类型进行定位
        if add_type is nn.modules.conv.Conv2d:
            if len(self.conv_li) == 0:
                return 1
            else:
                return eval(f'self.net.conv_pool.conv{idx - 1}.out_channels')
        if add_type is (nn.modules.pooling.AdaptiveMaxPool2d or nn.modules.pooling.AdaptiveAvgPool2d):
            pass
        if add_type is nn.modules.linear.Linear:
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
            kernel_size = np.random.choice(self.conv_args['kernel_size'])
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

    def mutate(self):
        # 这里变异只进行修改，不进行增删
        length = self.cp_num + self.fc_num
        if np.random.randint(0, length) < self.cp_num:
            mutate = np.random.choice(self.conv_li + self.pool_li[0:-1])  # 这里是为了避免选择到最后一层导致线性层发生变化
            module_type = mutate[0:-1]
            idx = int(mutate[-1])
            if module_type == 'conv':
                out_channels = np.random.choice(self.conv_args['out_channels'])
                # kernel_size = np.random.choice(range(1, self.cal_input_size(), 2))
                exec(f"self.net.conv_pool.{mutate}.out_channels = {out_channels}")
                if (module_type + str(idx+1)) in self.conv_li:
                    exec(f"self.net.conv_pool.{module_type + str(idx+1)}.in_channels"
                         f" = self.net.conv_pool.{mutate}.out_channels")
            elif module_type == 'pool':
                # 此处变异只使用MaxPooling
                output_size = np.random.choice(range(3, self.cal_input_size()))
                exec(f"self.net.conv_pool.{mutate}"
                     f" = nn.AdaptiveMaxPool2d((output_size, output_size))")
        else:
            mutate = np.random.choice(self.fc_li[0:-1])
            idx = int(mutate[-1])
            exec(f"self.net.linear.{mutate}.out_features = np.random.choice(self.line_args)")
            exec(f"self.net.linear.{mutate[0:-1] + str(idx+1)}.in_features = self.net.linear.{mutate}.out_features")

        self.net_format()

    def fitness_eval(self, train_loader, test_loader, epoch=20):
        # 训练模型
        with tqdm(total=epoch) as pbar:
            pbar.set_description(f'total parameters: {sum(p.numel() for p in self.net.parameters())}')
            for batch_idx, (data, target) in enumerate(train_loader):
                # 将数据转换为模型需要的格式
                if batch_idx >= epoch:
                    break
                data, target = data.cuda(), target.cuda()
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
                with torch.no_grad():
                    for data_test, target_test in test_loader:
                        # 将数据转换为模型需要的格式
                        data_test, target_test = data_test.cuda(), target_test.cuda()
                        # 前向传播
                        output = self.net(data_test)
                        # 获取预测结果
                        predict = output.argmax(dim=1)
                        # 计算准确率
                        correct += predict.eq(target_test.view_as(predict)).sum().item()
                        total += target_test.size(0)
                    accuracy = correct / total

                    eval_dic = {'Accuracy': accuracy}
                    pbar.set_postfix(eval_dic)
                pbar.update(1)
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

    def __init__(self, cp_num=9, fc_num=3, population_size=50) -> None:
        self.population = []
        self.init_population(cp_num, fc_num, population_size)

    def init_population(self, cp_num, fc_num, population_size):
        for idx in range(population_size):
            child = Child(cp_num, fc_num)
            individual = {'index': idx,
                          'individual': child,
                          'accuracy': 0}
            self.population.append(individual)

    def eval(self, train_loader, test_loader, epoch=20):
        for idx in range(len(self.population)):
            child = self.population[idx]['child']
            accuracy = child.fitness_eval(train_loader, test_loader, epoch)
            self.population[idx]['accuracy'] = accuracy

    def binary_tournament_select(self, non_elites, select_num):
        other = []
        compete = lambda competitors: max(competitors, key=competitors['accuracy'])
        for _ in range(select_num):
            parent = np.random.choice(non_elites, 2)
            other.append(compete(parent))
        return other

    def offspring_generate(self):
        offsprings = []
        mating_pool = self.environment_select()
        while len(mating_pool) > 0:
            parent1, parent2 = np.random.choice(mating_pool, 2)
            mating_pool.remove(parent1)
            mating_pool.remove(parent2)
            os1, os2 = self.crossover(parent1, parent2)
            os1.mutate()
            os2.mutate()
            offsprings.append(os1)
            offsprings.append(os2)
        return offsprings

    def crossover(self,
                  parent1: Child,
                  parent2: Child) -> (Child, Child):
        for idx in range(min(len(parent1.conv_li), len(parent2.conv_li))):
            if np.random.rand() > 0.5:
                exec(f"parent1.net.conv_pool.conv{idx}.out_channels, "
                     f"parent2.net.conv_pool.conv{idx}.out_channels = "
                     f"parent2.net.conv_pool.conv{idx}.out_channels, "
                     f"parent1.net.conv_pool.conv{idx}.out_channels")

        for idx in range(min(len(parent1.pool_li), len(parent2.pool_li))):
            if np.random.rand() > 0.5:
                exec(f"parent1.net.conv_pool.pool{idx}.output_size, "
                     f"parent2.net.conv_pool.pool{idx}.output_size = "
                     f"parent2.net.conv_pool.pool{idx}.output_size, "
                     f"parent1.net.conv_pool.pool{idx}.output_size")

        for idx in range(min(len(parent1.fc_li), len(parent2.fc_li))):
            if np.random.rand() > 0.5:
                exec(f"parent1.net.linear.linear{idx}.out_features, "
                     f"parent2.net.linear.linear{idx}.out_features = "
                     f"parent2.net.linear.linear{idx}.out_features, "
                     f"parent1.net.linear.linear{idx}.out_features")

        parent1.net_format()
        parent2.net_format()

    def environment_select(self):
        # 按照适应度从大到小排序
        sorted_pop = sorted(self.population, key=lambda x: x['accuracy'], reverse=True)
        # 选择20%精英个体
        elite_num = int(0.2 * len(self.population))
        elites = sorted_pop[:elite_num]
        # binary_tournament_select选择其他个体
        non_elites = self.binary_tournament_select(sorted_pop[elite_num:], len(self.population) - elite_num)
        # 返回精英个体和其他个体
        return elites + non_elites
