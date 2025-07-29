import torch
from torch import nn
import torch.nn.functional as F


class DGMU(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, residual=True, dilation_rate=6):
        super(DGMU, self).__init__()

        ops = [
            nn.Upsample(scale_factor=1, mode='trilinear', align_corners=True),
            nn.Conv3d(in_channels=n_filters_in, out_channels=n_filters_out, kernel_size=3, padding=1, stride=1),  # 常规卷积
            nn.BatchNorm3d(n_filters_out),  # 批量归一化
            nn.ReLU(inplace=True)  # ReLU 激活
        ]

        self.conv = nn.Sequential(*ops)

        attention_branch = [
            nn.Upsample(scale_factor=1, mode='trilinear', align_corners=True),  # 上采样
            nn.Conv3d(in_channels=n_filters_in, out_channels=n_filters_out, kernel_size=3, padding=dilation_rate,
                      stride=1, dilation=dilation_rate),  # 膨胀卷积
            nn.BatchNorm3d(n_filters_out),  # 批量归一化
            nn.ReLU(inplace=True)  # ReLU 激活
        ]
        self.side_conv = nn.Sequential(*attention_branch)

        self.residual_mode = residual

    def forward(self, x):
        y = self.conv(x)
        a = self.side_conv(x)

        if self.residual_mode:
            y = torch.sigmoid(a) * y + y
        else:
            y = torch.sigmoid(a) * y

        return y