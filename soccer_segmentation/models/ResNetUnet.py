from soccer_segmentation.models.decoder.unet_v2 import UNet as UNet_v2
from soccer_segmentation.models.decoder.unet_v3 import UNet as UNet_v3
from soccer_segmentation.models.encoder.resnet import *


class ResNet18UNet(nn.Module):
    def __init__(self, num_classes, train_encoder=False):
        super(ResNet18UNet, self).__init__()
        self.encoder = ResNet18(train_cnn=train_encoder)
        self.decoder = UNet_v2(num_classes)
        self.name = "ResNet18UNet"

    def forward(self, inputs):
        x = self.encoder(inputs)
        x = self.decoder(x)
        return x

    def unfreeze(self):
        self.encoder.unfreeze()


class ResNet34UNet(nn.Module):
    def __init__(self, num_classes, train_encoder=False):
        super(ResNet34UNet, self).__init__()
        self.encoder = ResNet34(train_cnn=train_encoder)
        self.decoder = UNet_v2(num_classes)
        self.name = "ResNet34UNet"

    def forward(self, inputs):
        x = self.encoder(inputs)
        x = self.decoder(x)
        return x

    def unfreeze(self):
        self.encoder.unfreeze()


class ResNet50UNet(nn.Module):
    def __init__(self, num_classes, train_encoder=False):
        super(ResNet50UNet, self).__init__()
        self.encoder = ResNet50(train_cnn=train_encoder)
        self.decoder = UNet_v3(num_classes)
        self.name = "ResNet50UNet"

    def forward(self, inputs):
        x = self.encoder(inputs)
        x = self.decoder(x)
        return x

    def unfreeze(self):
        self.encoder.unfreeze()


class ResNet101UNet(nn.Module):
    def __init__(self, num_classes, train_encoder=False):
        super(ResNet101UNet, self).__init__()
        self.encoder = ResNet101(train_cnn=train_encoder)
        self.decoder = UNet_v3(num_classes)
        self.name = "ResNet101UNet"

    def forward(self, inputs):
        x = self.encoder(inputs)
        x = self.decoder(x)
        return x

    def unfreeze(self):
        self.encoder.unfreeze()


class ResNet152UNet(nn.Module):
    def __init__(self, num_classes, train_encoder=False):
        super(ResNet152UNet, self).__init__()
        self.encoder = ResNet152(train_cnn=train_encoder)
        self.decoder = UNet_v3(num_classes)
        self.name = "ResNet152UNet"

    def forward(self, inputs):
        x = self.encoder(inputs)
        x = self.decoder(x)
        return x

    def unfreeze(self):
        self.encoder.unfreeze()


if __name__ == "__main__":
    model = ResNet101UNet(num_classes=3)
    print(model)
