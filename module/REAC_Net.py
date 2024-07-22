import torch
from module.resnet50 import resnet50
import torch.nn as nn
import torch.nn.functional as F
from module.deform_conv_v2 import DeformConv2d
from fightingcv_attention.attention.CBAM import CBAMBlock
from SoftPool import SoftPool2d



# -----------------------------------------#
#   ASPP特征提取模块
#   利用不同膨胀率的膨胀卷积进行特征提取
# -----------------------------------------#
class ASPP(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(ASPP, self).__init__()
        # 1×1卷积
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        # 3×3空洞卷积，d=6
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        # 3×3空洞卷积，d=12
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        # 3×3空洞卷积，d=18
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        # 1×1卷积，替代池化层操作，（卷积层可以进行一次选择，选择丢弃哪些信息且参数可训练）
        self.branch5_pool = SoftPool2d(kernel_size=1, stride=1)
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)

        # 对五个分支的结果concat（降低通道，特征融合）
        self.conv_cat = nn.Sequential(
            # DeformConv2d(dim_out*5, dim_out, 1, 0, 1, True),
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        [b, c, row, col] = x.size()
        # -----------------------------------------#
        #   一共五个分支
        # -----------------------------------------#
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        # -----------------------------------------#
        #   第五个分支，全局平均池化+卷积
        #   torch.mean()求平均，对应第五分支Avgpool，指定维度为2,3即H,W，并保证维度不发生改变
        # -----------------------------------------#
        # global_feature = torch.mean(x, 2, True)
        # global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_pool(x)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        # -----------------------------------------#
        # 对global_feature输入进行双线性上采样，保证最终的输出结果为输入ASPP的大小，并输出；
        # align_corners设置为True，输入和输出张量由其角像素的中心点对齐，从而保留角像素处的值。
        # -----------------------------------------#
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)

        # -----------------------------------------#
        #   将五个分支的内容堆叠起来
        #   然后1x1卷积整合特征。
        # -----------------------------------------#
        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        result = self.conv_cat(feature_cat)
        return result


# ----------------------------------#
#   SW和R50两个编码器的提取结果进行融合
# ----------------------------------#
class REAC_Net (nn.Module):
    def __init__(self, pretrained=True, channel_1=2048, num_classes=5, downsample_factor=16):
        super(REAC_Net, self).__init__()

        # ----------------------------------#
        #   对两个编码器进行定义
        # ----------------------------------#
        self.backbone_1 = resnet50(pretrained=pretrained)

        # 添加全局平均池化层
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 添加分类层
        self.classifier = nn.Linear(64, num_classes)

        # 为模型插入CBAM模块，防止分割结果特征图像椒盐现象
        self.cabm = CBAMBlock(channel=256, reduction=16, kernel_size=17)
        # -----------------------------------------#
        #   ASPP特征提取模块
        #   利用不同膨胀率的膨胀卷积进行特征提取
        # -----------------------------------------#
        self.aspp = ASPP(dim_in=channel_1, dim_out=256, rate=16 // downsample_factor)
        self.cls_conv_1 = nn.Conv2d(256, 64, 1, stride=1)

    def forward(self, x):
        # -----------------------------------------#
        #   获取CNN分支编码器提取结果
        # -----------------------------------------#
        lowfeature, x_1 = self.backbone_1(x)
        x_1 = self.aspp(x_1)
        x_1 = self.cabm(x_1)
        # x_1 = F.interpolate(x_1, size=(lowfeature.size(2)//4, lowfeature.size(3)//4),
        #                     mode='bilinear', align_corners=True)
        x_1 = self.cls_conv_1(x_1)

        # 全局平均池化
        x_1 = self.global_avg_pool(x_1)

        # 展平特征图
        x_1 = x_1.view(x_1.size(0), -1)  # 将 x_3 展平为 (batch_size, 64)

        # 分类
        x_1 = self.classifier(x_1)

        return x_1

if __name__ == '__main__':
    net = REAC_Net()
    input = torch.ones((2, 3, 512, 512))
    output = net(input)
    print(output.shape)
