import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from models.sync_batchnorm import SynchronizedBatchNorm2d

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ASPP_no4level(nn.Module):
    def __init__(self, backbone, output_stride, BatchNorm):
        super(ASPP_no4level, self).__init__()
        if backbone == 'drn':
            inplanes = 512
        elif backbone == 'mobilenet':
            inplanes = 320
        else:
            inplanes = 2048
            low_level_inplanes = 256 #
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1_128 = _ASPPModule(64, 64, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp1_256 = _ASPPModule(256, 64, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp1_1024 = _ASPPModule(1024, 128, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)

        self.bn1_128 = BatchNorm(64)
        self.bn1_256 = BatchNorm(64)
        self.bn1_1024 = BatchNorm(128)
        # self.bn1_2048 = BatchNorm(256)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

        self.last_conv = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                BatchNorm(256),
                                nn.ReLU(inplace=True),
                                nn.Dropout(0.5))

        self._init_weight()
        print("ASPP_4level")

    def forward(self, x_1, x_2, x_3):
        x_1 = self.aspp1_128(x_1)
        x_1 = self.bn1_128(x_1)
        x_1 = self.relu(x_1)
        x_1 = self.dropout(x_1)

        x_2 = self.aspp1_256(x_2)
        x_2 = self.bn1_256(x_2)
        x_2 = self.relu(x_2)
        x_2 = self.dropout(x_2)

        x_3 = self.aspp1_1024(x_3)
        x_3 = self.bn1_1024(x_3)
        x_3 = self.relu(x_3)
        x_3 = self.dropout(x_3)

        x_2 = F.interpolate(x_2, size=x_1.size()[2:], mode='bilinear', align_corners=True)
        x_3 = F.interpolate(x_3, size=x_1.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x_1, x_2, x_3), dim=1)
        x = self.last_conv(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
