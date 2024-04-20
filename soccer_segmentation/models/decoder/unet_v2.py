import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_op(x)


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        #print("UP")
        #print(x1.shape)
        x1 = self.up(x1)
        #print(x1.shape)
        #print(x2.shape)

        x = torch.cat([x1, x2], 1)
        #print(x.shape)
        #print(self.conv(x).shape)
        return self.conv(x)


class UpSampleEq(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = DoubleConv(2*in_channels, in_channels)
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], 1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.up_convolution_1 = UpSample(512, 256)
        self.up_convolution_2 = UpSample(256, 128)
        self.up_convolution_3 = UpSample(128, 64)
        self.up_convolution_4 = UpSampleEq(64)

        self.out = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)

    def forward(self, inputs):

        x0 = inputs[0]
        x1 = inputs[1]
        x2 = inputs[2]
        x3 = inputs[3]
        x4 = inputs[4]

        up_1 = self.up_convolution_1(x4, x3)
        up_2 = self.up_convolution_2(up_1, x2)
        up_3 = self.up_convolution_3(up_2, x1)
        up_4 = self.up_convolution_4(up_3, x0)

        out = self.out(up_4)
        return out
