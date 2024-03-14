from algorithm import *
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

# 加载数据集
train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

child = Child()
print(child)
conv = 'conv0'
exec(f'child.net.conv_pool.{conv}.out_channels = 18')
print(child)
accuracy = child.fitness_eval(train_loader, test_loader, epoch=1)

# ga = GA(20, 1, 200)
# ga.eval(train_loader, test_loader, epoch=1)