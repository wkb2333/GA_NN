from algorithm import *
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

# 加载数据集
train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# child = Child()
# print(child)
# print(child.fitness_eval(train_loader, test_loader, epoch=2))
# for _ in range(20):
#     child.mutate()
#
# print(child)
# print(child.fitness_eval(train_loader, test_loader, epoch=2))

ga = GA(cp_num=5, fc_num=2, population_size=20)
ga.evo(train_loader, test_loader, generation=5, epoch=1)
