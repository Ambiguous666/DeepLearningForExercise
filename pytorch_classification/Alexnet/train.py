import os
import sys
import json
import time

import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm

from model import AlexNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "val": transforms.Compose([transforms.Resize((224, 224)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
image_path = data_root + "/data_set/flower_data"
assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                     transform=data_transform["train"])  # 加载数据集
train_num = len(train_dataset)

# {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
flower_list = train_dataset.class_to_idx  # 获取不同分类的索引值
cla_dict = dict((val, key) for key, val in flower_list.items())
# write dict into json file
json_str = json.dumps(cla_dict, indent=4)  # 将分类的键值对转换成json格式
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

batch_size = 32
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                        transform=data_transform["val"])
val_num = len(validate_dataset)
validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=4, shuffle=False, num_workers=0)

# test_data_iter = iter(validate_loader)
# test_image, test_label = test_data_iter.next()
#
# def imshow(img):
#     img = img / 2 + 0.5  # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
#
# print(' '.join('%5s' % cla_dict[test_label[j].item()] for j in range(4)))
# imshow(utils.make_grid(test_image))

net = AlexNet(num_classes=5, init_weights=True)
net.to(device)

loss_function = nn.CrossEntropyLoss()
# pata = list(net.parameters())
optimizer = optim.Adam(net.parameters(), lr=0.0002)

save_path = './AlexNet.pth'

batch_acc = 0.0  # 最佳准确率，保存准确率最高的模型
for epoch in range(10):
    # train
    net.train()  # 通过调用net.train()会调用drop方法
    running_loss = 0.0  # 训练过程中的平均损失
    t1 = time.perf_counter()  # 计算训练一个epoch所用的时间
    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        optimizer.zero_grad()
        outputs = net(images.to(device))
        loss = loss_function(outputs, labels.to(device))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        rate = (step+1)/len(train_loader)  # len(train_loader)可以得到训练一轮所需要的步数
        a = "*" * int(rate*50)
        b = "." * int((1-rate)*50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate*100), a, b, loss), end="")
    print()
    print(time.perf_counter()-t1)

    # validate
    net.eval()
    acc = 0.0
    with torch.no_grad():  # 要求pytorch不对参数进行跟踪，在验证的过程中不计算损失梯度
        for data_test in validate_loader:  # 遍历验证集
            test_images, test_labels = data_test
            outputs = net(test_images.to(device))  # 网络的输出
            predict_y = torch.max(outputs, dim=1)  # dim=1表示从channel开始展平处理，保留了batch的维度 最大的预测值
            acc += (predict_y == test_labels.to(device)).sum().item()  # 计算准确率，正确为1，错误为0，最后累加
        accurate_test = acc / val_num  # 验证正确样本个数 / 总个数
        if accurate_test > best_acc:  # 如果当前准确率大于历史最优准确率
            best_acc = accurate_test  # 替换历史最佳准确率
            torch.save(net.state_dict(), save_path)  # 更新参数
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / step, acc / val_num))

print("Finished Training")

