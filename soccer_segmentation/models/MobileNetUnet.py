from soccer_segmentation.models.decoder.unet_v4 import UNet as UNet_v4
from soccer_segmentation.models.decoder.unet_v5 import UNet as UNet_v5
from soccer_segmentation.models.encoder.mobilenet import *


class MobileNetV3SmallUNet(nn.Module):
    def __init__(self, num_classes, train_encoder=False):
        super(MobileNetV3SmallUNet, self).__init__()
        self.encoder = MobileNetV3Small(train_cnn=train_encoder)
        self.decoder = UNet_v4(num_classes)
        self.name = "MobileNetV3SmallUNet"
        self.small_mask = True

    def forward(self, inputs):
        x = self.encoder(inputs)
        x = self.decoder(x)
        return x

    def unfreeze(self):
        self.encoder.unfreeze()


class MobileNetV3LargeUNet(nn.Module):
    def __init__(self, num_classes, train_encoder=False):
        super(MobileNetV3LargeUNet, self).__init__()
        self.encoder = MobileNetV3Large(train_cnn=train_encoder)
        self.decoder = UNet_v5(num_classes)
        self.name = "MobileNetV3LargeUNet"
        self.small_mask = True

    def forward(self, inputs):
        x = self.encoder(inputs)
        x = self.decoder(x)
        return x

    def unfreeze(self):
        self.encoder.unfreeze()


if __name__ == "__main__":
    model = MobileNetV3SmallUNet(num_classes=3)
    print(model)
