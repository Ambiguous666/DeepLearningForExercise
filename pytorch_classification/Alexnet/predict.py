import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import AlexNet

data_transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# load image
img = Image.open("../tulip.jpg")
plt.imshow(img)
# [N,C,H,W]
img = data_transform(img)
# expend batch dimension 把图片扩充一个维度 添加一个batch维度
img = torch.unsqueeze(img, dim=0)

# read class_indict
try:
    json_file = open('./class_indices.json', 'r')
    class_indict = json.load(json_file)
except Exception as e:
    print(e)
    exit(1)

# create model
model = AlexNet(num_classes=5)
# load model weights
model_weight_path = "./AlexNet.pth"
model.load_state_dict(torch.load(model_weight_path))
model.eval()  # 关闭dropout方法
with torch.no_grad():
    # predict class
    output = torch.squeeze(model(img))  # 把batch维度进行压缩
    predict = torch.softmax(output, dim=0)  # softmax变成概率分布
    predict_cla = torch.argmax(predict).numpy()
print(class_indict[str(predict_cla)], predict[predict_cla].item())
plt.show()
