from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegNet(nn.Module):

    def __init__(self, out_chn=3, momentum=0.5):
        super(SegNet, self).__init__()

        self.out_chn = out_chn
        self.MaxEn = nn.MaxPool2d(2, stride=2, return_indices=True)

        self.MaxDe = nn.MaxUnpool2d(2, stride=2)

        self.ConvDe53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDe53 = nn.BatchNorm2d(512, momentum=momentum)
        self.ConvDe52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDe52 = nn.BatchNorm2d(512, momentum=momentum)
        self.ConvDe51 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.BNDe51 = nn.BatchNorm2d(256, momentum=momentum)

        self.ConvDe43 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNDe43 = nn.BatchNorm2d(256, momentum=momentum)
        self.ConvDe42 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNDe42 = nn.BatchNorm2d(256, momentum=momentum)
        self.ConvDe41 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.BNDe41 = nn.BatchNorm2d(128, momentum=momentum)

        self.ConvDe33 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.BNDe33 = nn.BatchNorm2d(128, momentum=momentum)
        self.ConvDe32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.BNDe32 = nn.BatchNorm2d(128, momentum=momentum)
        self.ConvDe31 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.BNDe31 = nn.BatchNorm2d(64, momentum=momentum)

        self.ConvDe22 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.BNDe22 = nn.BatchNorm2d(64, momentum=momentum)
        self.ConvDe21 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.BNDe21 = nn.BatchNorm2d(64, momentum=momentum)

        self.ConvDe12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.BNDe12 = nn.BatchNorm2d(64, momentum=momentum)
        self.ConvDe11 = nn.Conv2d(64, self.out_chn, kernel_size=3, padding=1)
        self.BNDe11 = nn.BatchNorm2d(self.out_chn, momentum=momentum)

    def forward(self, inputs: List):

        x0 = inputs[0]
        x1 = inputs[1]
        x2 = inputs[2]
        x3 = inputs[3]
        x4 = inputs[4]

        x0, ind0 = self.MaxEn(x0)
        size0 = x0.size()
        x1, ind1 = self.MaxEn(x1)
        size1 = x1.size()
        x2, ind2 = self.MaxEn(x2)
        size2 = x2.size()
        x3, ind3 = self.MaxEn(x3)
        size3 = x3.size()
        x4, ind4 = self.MaxEn(x4)

        x = self.MaxDe(x4, ind4, output_size=size3)
        x = F.relu(self.BNDe53(self.ConvDe53(x)))
        x = F.relu(self.BNDe52(self.ConvDe52(x)))
        x = F.relu(self.BNDe51(self.ConvDe51(x)))

        x = self.MaxDe(x, ind3, output_size=size2)
        x = F.relu(self.BNDe43(self.ConvDe43(x)))
        x = F.relu(self.BNDe42(self.ConvDe42(x)))
        x = F.relu(self.BNDe41(self.ConvDe41(x)))

        x = self.MaxDe(x, ind2, output_size=size1)
        x = F.relu(self.BNDe33(self.ConvDe33(x)))
        x = F.relu(self.BNDe32(self.ConvDe32(x)))
        x = F.relu(self.BNDe31(self.ConvDe31(x)))

        x = self.MaxDe(x, ind1, output_size=size0)
        x = F.relu(self.BNDe22(self.ConvDe22(x)))
        x = F.relu(self.BNDe21(self.ConvDe21(x)))

        x = self.MaxDe(x, ind0)
        x = F.relu(self.BNDe12(self.ConvDe12(x)))
        x = self.ConvDe11(x)

        x = F.softmax(x, dim=1)

        return x


if __name__ == "__main__":
    model = SegNet()
    print(model)
