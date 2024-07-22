import torch
import torchvision
from PIL import Image
from torch import nn

image_path = "classification/train/017.jpg"
image = Image.open(image_path)
# print(image)
image = image.convert('RGB')

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((512, 512)),
                                            torchvision.transforms.ToTensor()])

image = transform(image)
# print(image.shape)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# module = torch.load("zz_test_30_gpu.pth", map_location=device)
module = torch.load("save_model/class/class_072201_10.pth", map_location=device)
# print(module)


image = image.to(device)

image = torch.reshape(image, (1, 3, 512, 512))
module.eval()
with torch.no_grad():
    output = module(image)
# print(output)
# print(output.argmax(1))


# 定义一个列表，存储物体种类
classes = ['类别1', '类别2', '类别3', '类别4', '类别5']

# 应用softmax函数获取概率
probabilities = torch.nn.functional.softmax(output, dim=1)

# 获取最大概率值和对应的索引
max_prob, num = probabilities.max(1)

print(f"图片对应各类别的概率分别是：{probabilities}")
print(f"图片中的可能是: {classes[num.item()]}, 概率为: {max_prob.item()}")