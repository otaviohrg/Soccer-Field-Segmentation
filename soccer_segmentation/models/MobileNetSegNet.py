from soccer_segmentation.models.decoder.segnet_v4 import SegNet as SegNet_v4
from soccer_segmentation.models.decoder.segnet_v5 import SegNet as SegNet_v5
from soccer_segmentation.models.encoder.mobilenet import *


class MobileNetV3SmallSegNet(nn.Module):
    def __init__(self, num_classes, train_encoder=False):
        super(MobileNetV3SmallSegNet, self).__init__()
        self.encoder = MobileNetV3Small(train_cnn=train_encoder)
        self.decoder = SegNet_v4(num_classes)
        self.name = "MobileNetV3SmallSegNet"

    def forward(self, inputs):
        x = self.encoder(inputs)
        x = self.decoder(x)
        return x

    def unfreeze(self):
        self.encoder.unfreeze()


class MobileNetV3LargeSegNet(nn.Module):
    def __init__(self, num_classes, train_encoder=False):
        super(MobileNetV3LargeSegNet, self).__init__()
        self.encoder = MobileNetV3Large(train_cnn=train_encoder)
        self.decoder = SegNet_v5(num_classes)
        self.name = "MobileNetV3SmallSegNet"

    def forward(self, inputs):
        x = self.encoder(inputs)
        x = self.decoder(x)
        return x

    def unfreeze(self):
        self.encoder.unfreeze()


if __name__ == "__main__":
    model = MobileNetV3SmallSegNet(num_classes=3)
    print(model)
