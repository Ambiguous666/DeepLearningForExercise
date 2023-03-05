import torch
import torchvision
import torch.nn as nn
from model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


# 对图像进行预处理
transforms = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])

# 50000张训练图片
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False,
                                        transform=transforms)

# 每一批36张图片 shuffle是对图片进行打乱
trainloader = torch.utils.data.DataLoader(trainset, batch_size=36,
                                          shuffle=True, num_workers=0)

# 10000张测试图片
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transforms)

testloader = torch.utils.data.DataLoader(testset, batch_size=10000,
                                         shuffle=False, num_workers=0)

# 使用迭代器
test_data_iter = iter(testloader)
test_image, test_label = test_data_iter.next()

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = LeNet()
loss_function = nn.CrossEntropyLoss()  # 损失函数 这里面已经包含了softmax函数
optimizer = optim.Adam(net.parameters(),lr=0.001)  # 优化器

for epoch in range(5):  # 对训练集训练多少轮
    running_loss = 0.0  # 累加在训练中的损失
    for step, data in enumerate(trainloader, start=0):
        inputs, lables = data
        optimizer.zero_grad()  # 清除历史梯度
        outputs = net(inputs)
        loss = loss_function(outputs, lables)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if step % 500 == 499:  # step表示某一轮的多少步
            with torch.no_grad():
                outputs = net(test_image)
                predict_y = torch.max(outputs, dim=1)[1]
                accuracy = (predict_y == test_label).sum().item() / test_label.size(0)

                print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f'
                      % (epoch + 1, step + 1, running_loss / 500, accuracy))
                running_loss = 0.0

print('Finished Training')
save_path = './Lenet.pth'
torch.save(net.state_dict(), save_path)




