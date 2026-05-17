import torch
import torch.nn as nn
import torch.nn.functional as F
# from ResNet import resnet50
from math import log

from .decoder_m import weight_init


class ConvBNR(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(ConvBNR, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size, stride=stride, padding=dilation, dilation=dilation, bias=bias),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class Conv1x1(nn.Module):
    def __init__(self, inplanes, planes):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class EAM(nn.Module):
    def __init__(self):
        super(EAM, self).__init__()
        self.reduce1 = Conv1x1(64, 64)
        self.reduce4 = Conv1x1(512, 256)
        self.block = nn.Sequential(
            ConvBNR(256 + 64, 256, 3),
            ConvBNR(256, 256, 3),
            nn.Conv2d(256, 1, 1))

    def forward(self, x4, x1):
        size = x1.size()[2:]
        x1 = self.reduce1(x1)
        x4 = self.reduce4(x4)
        x4 = F.interpolate(x4, size, mode='bilinear', align_corners=False)
        out = torch.cat((x4, x1), dim=1)
        out = self.block(out)

        return out




class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
    def forward(self, x):
        avg_out = self.mlp(F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))))
        max_out = self.mlp(F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))))
        channel_att_sum = avg_out + max_out

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.spatial = nn.Conv2d(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2)
    def forward(self, x):
        x_compress = torch.cat((torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale



class EFM(nn.Module):
    def __init__(self, channel, reduction_ratio=16):
        super(EFM, self).__init__()
        # 原始EFM组件
        t = int(abs((log(channel, 2) + 1) / 2))
        k = t if t % 2 else t + 1
        self.conv2d = ConvBNR(channel, channel, 3)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
        # 双注意力门控（用于增强权重计算）
        self.channel_gate = ChannelGate(channel, reduction_ratio)
        self.spatial_gate = SpatialGate()

    def forward(self, c, att):
        """
        输入:
        - c: 特征图 [N, C, H, W]
        - att: 注意力图 [N, C, H, W] 或不同分辨率
        """
        # 调整注意力图分辨率 (如果需要)
        if c.size() != att.size():
            att = F.interpolate(att, size=c.size()[2:], 
                                mode='bilinear', 
                                align_corners=False)
        
        # 特征调制
        x = c * att + c  # 特征调制
        x = self.conv2d(x)  # 卷积处理
        
        # 通道缩放
        wei = self.avg_pool(x)  # 全局平均池化 [N, C, 1, 1]
        wei = wei.squeeze(-1)    # 移除W维度 [N, C, 1]
        wei = wei.transpose(1, 2)  # 转置为 [N, 1, C]
        
        wei = self.conv1d(wei)   # 1D卷积处理 [N, 1, C]
        wei = wei.transpose(1, 2)  # 转置回 [N, C, 1]
        wei = wei.unsqueeze(-1)  # 添加W维度 [N, C, 1, 1]
        # wei = self.sigmoid(wei)
        xe1 = self.channel_gate(x)
        # xe2 = self.spatial_gate(x)
        # x = xe

        return xe1*wei  # [N, C, H, W]

