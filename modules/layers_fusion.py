import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


###################################################
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


###################################################
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


###################################################
class CrossAttention(nn.Module):
    def __init__(self, channel):
        super(CrossAttention, self).__init__()
        self.conv_x1 = nn.Conv2d(channel * 2, channel * 2, kernel_size=1, padding=0, bias=False)
        self.conv_x2 = nn.Conv2d(channel * 2, channel, kernel_size=1, padding=0, bias=False)
        self.sigmoid_x = nn.Sigmoid()

        self.conv_y1 = nn.Conv2d(channel * 2, channel * 2, kernel_size=1, padding=0, bias=False)
        self.conv_y2 = nn.Conv2d(channel * 2, channel, kernel_size=1, padding=0, bias=False)
        self.sigmoid_y = nn.Sigmoid()

    def forward(self, x, y):
        out = torch.cat([x, y], dim=1)
        out_x = self.sigmoid_x(self.conv_x2(self.conv_x1(out)))
        out_y = self.sigmoid_y(self.conv_y2(self.conv_y1(out)))
        return out_x, out_y


###################################################
class cbam_block(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=7):
        super(cbam_block, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        return x


##################################################################
class AFF(nn.Module):
    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        xa = x + y
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = 2 * x * wei + 2 * y * (1 - wei)
        return xo


###################################################
class Sobelxy(nn.Module):
    def __init__(self, channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(Sobelxy, self).__init__()
        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])
        self.convx = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride,
                               dilation=dilation, groups=channels, bias=False)
        self.convx.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.convy = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride,
                               dilation=dilation, groups=channels, bias=False)
        self.convy.weight.data.copy_(torch.from_numpy(sobel_filter.T))

    def forward(self, x):
        sobelx = self.convx(x)
        sobely = self.convy(x)
        x = torch.abs(sobelx) + torch.abs(sobely)
        return x


###################################################
class Inception(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Inception, self).__init__()
        self.branch0 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=1),
                                     nn.BatchNorm2d(in_channels),
                                     nn.LeakyReLU()
                                     )
        self.branch1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(in_channels),
                                     nn.LeakyReLU()
                                     )
        self.branch2 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2),
                                     nn.BatchNorm2d(in_channels),
                                     nn.LeakyReLU()
                                     )
        self.conv = nn.Conv2d(in_channels * 4, out_channels, kernel_size=1)
        self.sobel = Sobelxy(in_channels)
        self.grid_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.batchnorm = nn.BatchNorm2d(in_channels)
        self.PReLU = nn.PReLU()

    def forward(self, x):
        b0 = self.branch0(x)
        b1 = self.branch1(x)
        b2 = self.branch2(x)

        grid = self.sobel(x)
        grid = self.grid_conv(grid)

        # b = torch.cat([b1, b2, b3, grid], dim=1)
        # b = torch.cat([x, b1, b2, b3], dim=1)
        b = torch.cat([x,b0, b1, b2], dim=1)
        b = self.conv(b)

        return b


###################################################
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.stride = stride

        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Add the shortcut connection
        out += self.shortcut(x)
        out = self.relu(out)

        return out


###################################################
class Inception_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Inception_ResidualBlock, self).__init__()
        self.inception1 = Inception(in_channels, out_channels)
        self.inception2 = Inception(out_channels, out_channels)
        self.relu = nn.ReLU()
        self.stride = stride
        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        residual = x
        out = self.inception1(x)
        out = self.inception2(out)

        # Add the shortcut connection
        out += self.shortcut(x)
        out = self.relu(out)

        return out


##################################################
class ConvBnLeakyRelu2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                              dilation=dilation, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.leaky_relu(self.conv(x), negative_slope=0.2)


###################################################
class ConvBnTanh2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnTanh2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                              dilation=dilation, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return torch.tanh(self.conv(x)) / 2 + 0.5


###################################################
class ConvLeakyRelu2d(nn.Module):
    # convolution
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                              dilation=dilation, groups=groups)
        # self.bn   = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # print(x.size())
        return F.leaky_relu(self.conv(x), negative_slope=0.2)


if __name__ == '__main__':
    # x = torch.randn([4, 64, 128, 128])
    # Inc = Inception(64)
    # y = Inc(x)
    # print(y.shape)

    res_block = ResidualBlock(in_channels=64, out_channels=64)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    res_block.to(device)

    x = torch.randn(4, 64, 256, 256).to(device)  # Assuming batch size=4, channel=64, height=64, width=64
    output = res_block(x)
    print(output.shape)  # Check the output shape