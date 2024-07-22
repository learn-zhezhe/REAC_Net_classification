import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
from module.REAC_Net import REAC_Net
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


# 超参数设置
# -----------------------------------------
batch_size = 4
lr = 5e-4
wd = 1e-5
epoch = 50
# -----------------------------------------


# -----------------------------------------
# 加载自己的数据集的时候可以使用
# -----------------------------------------
class MVTecDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [os.path.join(root, f) for root, _, files in os.walk(root_dir) for f in files if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image = Image.open(img_name)

        # 从文件路径中提取标签
        label_name = os.path.basename(os.path.dirname(img_name))
        label = {
            '类别1': 0,
            '类别2': 1,
            '类别3': 2,
            '类别4': 3,
            '类别5': 4
        }.get(label_name, 0)  # 默认为“good”

        if self.transform:
            image = self.transform(image)

        return image, label

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((512, 512)),  # 根据MVTec_AD数据集的图像尺寸调整
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# 初始化数据集和数据加载器
train_dataset = MVTecDataset(root_dir='classification/train', transform=transform)
val_dataset = MVTecDataset(root_dir='classification/val', transform=transform)
train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_data = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 定义训练设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# length长度
train_data_size = len(train_dataset)
test_data_size = len(val_dataset)
# 如果train_data_size为10，训练数据集的长度为：10
print(f"训练数据集的长度为：{train_data_size}" )
print("测试数据集的长度为：%d" % test_data_size)


# 模型的训练
# net = STRD()
net = REAC_Net()
net = net.to(device)

# 定义Focal Loss损失函数
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction=self.reduction)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss


# 损失函数
loss_fn = nn.CrossEntropyLoss()
# loss_fn = FocalLoss()
loss_fn = loss_fn.to(device)

# 优化器
# optimizer = torch.optim.SGD(net.parameters(), lr=lr)
optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=wd)

# 添加tensorboard
writer = SummaryWriter("logs_class")

for i in range(epoch):
    print(f"------第{i+1}轮训练开始------")

    # 训练步骤开始
    train_loss = 0.0  # 用于累积训练损失
    correct_train_preds = 0.0  # 用于累积训练集上正确的预测数
    total_train_samples = 0.0  # 用于记录训练集样本总数

    net.train()  # 模型设置为训练模式
    for data in train_data:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = net(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 累积损失和正确预测数
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        correct_train_preds += predicted.eq(targets).sum().item()
        total_train_samples += targets.size(0)

    # 计算平均损失和准确率
    avg_train_loss = train_loss / train_data_size
    train_accuracy = correct_train_preds / train_data_size

    # 打印训练集上的损失和精度
    print(f"训练集上的Loss：{avg_train_loss:.4f}")
    print(f"训练集上的正确率：{train_accuracy:.4f}")

    # 测试步骤开始
    test_loss = 0.0  # 用于累积测试损失
    correct_test_preds = 0.0  # 用于累积测试集上正确的预测数

    net.eval()  # 模型设置为评估模式
    with torch.no_grad():
        for data in val_data:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = net(imgs)
            loss = loss_fn(outputs, targets)

            # 累积损失和正确预测数
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            correct_test_preds += predicted.eq(targets).sum().item()

    # 计算平均损失和准确率
    avg_test_loss = test_loss / test_data_size
    test_accuracy = correct_test_preds / test_data_size

    # 打印测试集上的损失和精度
    print(f"验证集上的Loss：{avg_test_loss:.4f}")
    print(f"验证集上的正确率：{test_accuracy:.4f}")

    # 使用TensorBoard记录训练和测试损失及准确率
    writer.add_scalar("train_loss", avg_train_loss, i + 1)
    writer.add_scalar("train_accuracy", train_accuracy, i + 1)
    writer.add_scalar("test_loss", avg_test_loss, i + 1)
    writer.add_scalar("test_accuracy", test_accuracy, i + 1)

    # # 每次迭代后增加测试步骤计数器
    # total_test_step += 1

    # if i > 20 and i % 5 == 0:
    if (i+1) % 2 == 0:
        module = net
        folder_path = 'save_model/class'
        os.makedirs(folder_path, exist_ok=True)
        file_name = 'class_072201_{}.pth'.format(i + 1)
        file_path = folder_path + '/' +file_name
        # 模型储存方式一 ->> 完整模型储存
        torch.save(module, file_path)
        # torch.save(net, "class_072101_{}_gpu.pth".format(i+1))
        # torch.save(zz_test.state_dict(), "zz_test_{}.pth".format(i))
        print("模型已保存")

writer.close()