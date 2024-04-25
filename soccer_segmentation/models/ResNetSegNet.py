from soccer_segmentation.models.decoder.segnet_v2 import SegNet as SegNet_v2
from soccer_segmentation.models.decoder.segnet_v3 import SegNet as SegNet_v3
from soccer_segmentation.models.encoder.resnet import *


class ResNet18SegNet(nn.Module):
    def __init__(self, num_classes, train_encoder=False, momentum=0.5):
        super(ResNet18SegNet, self).__init__()
        self.encoder = ResNet18(train_cnn=train_encoder)
        self.decoder = SegNet_v2(num_classes, momentum=momentum)
        self.name = "ResNet18SegNet"
        self.small_mask = True

    def forward(self, inputs):
        x = self.encoder(inputs)
        x = self.decoder(x)
        return x

    def unfreeze(self):
        self.encoder.unfreeze()


class ResNet34SegNet(nn.Module):
    def __init__(self, num_classes, train_encoder=False, momentum=0.5):
        super(ResNet34SegNet, self).__init__()
        self.encoder = ResNet34(train_cnn=train_encoder)
        self.decoder = SegNet_v2(num_classes, momentum=momentum)
        self.name = "ResNet34SegNet"
        self.small_mask = True

    def forward(self, inputs):
        x = self.encoder(inputs)
        x = self.decoder(x)
        return x

    def unfreeze(self):
        self.encoder.unfreeze()


class ResNet50SegNet(nn.Module):
    def __init__(self, num_classes, train_encoder=False, momentum=0.5):
        super(ResNet50SegNet, self).__init__()
        self.encoder = ResNet50(train_cnn=train_encoder)
        self.decoder = SegNet_v3(num_classes, momentum=momentum)
        self.name = "ResNet50SegNet"
        self.small_mask = True

    def forward(self, inputs):
        x = self.encoder(inputs)
        x = self.decoder(x)
        return x

    def unfreeze(self):
        self.encoder.unfreeze()


class ResNet101SegNet(nn.Module):
    def __init__(self, num_classes, train_encoder=False, momentum=0.5):
        super(ResNet101SegNet, self).__init__()
        self.encoder = ResNet101(train_cnn=train_encoder)
        self.decoder = SegNet_v3(num_classes, momentum=momentum)
        self.name = "ResNet101SegNet"
        self.small_mask = True

    def forward(self, inputs):
        x = self.encoder(inputs)
        x = self.decoder(x)
        return x

    def unfreeze(self):
        self.encoder.unfreeze()


class ResNet152SegNet(nn.Module):
    def __init__(self, num_classes, train_encoder=False, momentum=0.5):
        super(ResNet152SegNet, self).__init__()
        self.encoder = ResNet152(train_cnn=train_encoder)
        self.decoder = SegNet_v3(num_classes, momentum=momentum)
        self.name = "ResNet152SegNet"
        self.small_mask = True

    def forward(self, inputs):
        x = self.encoder(inputs)
        x = self.decoder(x)
        return x

    def unfreeze(self):
        self.encoder.unfreeze()


if __name__ == "__main__":
    model = ResNet101SegNet(num_classes=3)
    print(model)
