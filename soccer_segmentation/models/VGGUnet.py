import torch.nn as nn
from soccer_segmentation.models.decoder.unet_v1 import UNet
from soccer_segmentation.models.encoder.vgg import VGG11, VGG13, VGG16, VGG19


class VGG11UNet(nn.Module):
    def __init__(self, num_classes, name="VGG11Unet", train_encoder=False):
        super(VGG11UNet, self).__init__()
        self.encoder = VGG11(train_cnn=train_encoder)
        self.decoder = UNet(num_classes)
        self.name = name

    def forward(self, inputs):
        x = self.encoder(inputs)
        x = self.decoder(x)
        return x

    def unfreeze(self):
        self.encoder.unfreeze()


class VGG13UNet(nn.Module):
    def __init__(self, num_classes, name="VGG13Unet", train_encoder=False):
        super(VGG13UNet, self).__init__()
        self.encoder = VGG13(train_cnn=train_encoder)
        self.decoder = UNet(num_classes)
        self.name = name

    def forward(self, inputs):
        x = self.encoder(inputs)
        x = self.decoder(x)
        return x

    def unfreeze(self):
        self.encoder.unfreeze()


class VGG16UNet(nn.Module):
    def __init__(self, num_classes, name="VGG16Unet", train_encoder=False):
        super(VGG16UNet, self).__init__()
        self.encoder = VGG16(train_cnn=train_encoder)
        self.decoder = UNet(num_classes)
        self.name = name

    def forward(self, inputs):
        x = self.encoder(inputs)
        x = self.decoder(x)
        return x

    def unfreeze(self):
        self.encoder.unfreeze()


class VGG19UNet(nn.Module):
    def __init__(self, num_classes, name="VGG19Unet", train_encoder=False):
        super(VGG19UNet, self).__init__()
        self.encoder = VGG19(train_cnn=train_encoder)
        self.decoder = UNet(num_classes)
        self.name = name

    def forward(self, inputs):
        x = self.encoder(inputs)
        x = self.decoder(x)
        return x

    def unfreeze(self):
        self.encoder.unfreeze()


if __name__ == "__main__":
    model = VGG11UNet(num_classes=3)
    print(model)
