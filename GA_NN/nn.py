import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from tqdm import tqdm

# 加载数据集
'''
关于MNIST数据集：
  train:test = 60000:10000
  0-9手写数字 28*28带标签灰度图像
'''
train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 定义模型的搜索空间
filters = [16, 32, 64] # 卷积核数目
kernels = [3, 5] # 卷积核大小
poolings = [2, 3] # 池化层大小
units = [64, 128, 256] # 全连接层神经元数目
activations = ['relu', 'sigmoid'] # 激活函数
dropout_rates = [0.1, 0.2, 0.3] # Dropout率

# 定义模型评估函数
def evaluate_model(model, loss_func, optimizer):
  # 训练模型
  for epoch in range(10):
    pbar = tqdm()
    for batch_idx, (data, target) in enumerate(train_loader):
      # 将数据转换为模型需要的格式
      # data, target = data.cuda(), target.cuda()
      # 前向传播
      output = model(data)
      # 计算损失
      loss = loss_func(output, target)
      # 反向传播
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      pbar.update(1)
  # 评估模型
  correct = 0
  total = 0
  for data, target in test_loader:
    # 将数据转换为模型需要的格式
    # data, target = data.cuda(), target.cuda()
    # 前向传播
    output = model(data)
    # 获取预测结果
    pred = output.max(1, keepdim=True)[1]
    # 计算准确率
    correct += pred.eq(target.view_as(pred)).sum().item()
    total += target.size(0)
  accuracy = correct / total
  # 返回准确率
  return accuracy

# 定义种群初始化函数
def initial_population():
  population = []
  for i in range(20):
    # 随机生成一条染色体，代表一个模型
    chromosome = {
      'filters': np.random.choice(filters),
      'kernel': np.random.choice(kernels),
      'pooling': np.random.choice(poolings),
      'units': np.random.choice(units),
      'activation': np.random.choice(activations),
      'dropout_rate': np.random.choice(dropout_rates)
    }
    # 将染色体加入种群
    population.append(chromosome)
  return population

# 定义选择函数
def selection(population):
  # 将种群按照染色体对应的准确率从大到小排序
  sorted_pop = sorted(population, key=lambda x: x['accuracy'], reverse=True)
  # 计算精英个体数量
  elite_num = int(0.2 * 20)
  # 选择精英个体
  elites = sorted_pop[:elite_num]
  # 随机选择其他个体
  non_elites = np.random.choice(sorted_pop[elite_num:], 20 - elite_num, replace=False)
  # 返回精英个体和其他个体
  return elites, non_elites

# 定义交叉函数
def crossover(elites):
  # 从精英个体中随机选择两个染色体
  parent1 = np.random.choice(elites)
  parent2 = np.random.choice(elites)
  # 产生新的染色体
  child = {
    'filters': np.random.choice([parent1['filters'], parent2['filters']]),
    'kernel': np.random.choice([parent1['kernel'], parent2['kernel']]),
    'pooling': np.random.choice([parent1['pooling'], parent2['pooling']]),
    'units': np.random.choice([parent1['units'], parent2['units']]),
    'activation': np.random.choice([parent1['activation'], parent2['activation']]),
    'dropout_rate': np.random.choice([parent1['dropout_rate'], parent2['dropout_rate']])
  }
  return child

# 定义变异函数
def mutation(child):
  # 随机选择一个基因进行变异
  gene = np.random.choice(list(child.keys()))
  # 根据基因不同进行相应变异操作
  if gene == 'filters':
    child[gene] = np.random.choice(filters)
  elif gene == 'kernel':
    child[gene] = np.random.choice(kernels)
  elif gene == 'pooling':
    child[gene] = np.random.choice(poolings)
  elif gene == 'units':
    child[gene] = np.random.choice(units)
  elif gene == 'activation':
    child[gene] = np.random.choice(activations)
  else:
    child[gene] = np.random.choice(dropout_rates)
  return child

# 定义遗传算法主函数
def genetic_algorithm():
  # 初始化种群
  population = initial_population()
  # 定义迭代次数
  for epoch in range(50):
    # 评估种群中每个个体的准确率
    for chromosome in population:
      # 构建模型
      model = nn.Sequential(
        nn.Conv2d(1, chromosome['filters'], chromosome['kernel']),
        nn.ReLU() if chromosome['activation'] == 'relu' else nn.Sigmoid(),
        nn.MaxPool2d(chromosome['pooling']),
        nn.Flatten(),
        # nn.Linear(chromosome['filters'] * 13 * 13, chromosome['units']),
        nn.Linear(4096, chromosome['units']),
        nn.ReLU() if chromosome['activation'] == 'relu' else nn.Sigmoid(),
        nn.Dropout(chromosome['dropout_rate']),
        nn.Linear(chromosome['units'], 10),
        nn.LogSoftmax(dim=1)
      )
      # 将模型转移到GPU上
      # model.cuda()
      # 定义损失函数
      loss_func = nn.NLLLoss()
      # 定义优化器
      optimizer = optim.Adam(model.parameters(), lr=0.001)
      # 计算准确率
      chromosome['accuracy'] = evaluate_model(model, loss_func, optimizer)
    # 进行选择
    elites, non_elites = selection(population)
    # 进行交叉和变异操作，并将新个体加入种群
    for i in range(10):
      population.append(crossover(elites))
      population.append(mutation(np.random.choice(elites)))
    # 打印每一代中最优个体的准确率
    print("Generation {}: Best accuracy is {}".format(epoch + 1, max([chromosome['accuracy'] for chromosome in population])))
  # 返回最优个体
  return max(population, key=lambda x: x['accuracy'])

# 运行遗传算法
best_model = genetic_algorithm()

# 打印最优模型的准确率和参数
print("Best accuracy is {}, with the following parameters: {}".format(best_model['accuracy'], best_model))