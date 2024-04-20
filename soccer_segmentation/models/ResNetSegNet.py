import torch.nn as nn
from soccer_segmentation.models.decoder.segnet_v2 import SegNet
from soccer_segmentation.models.encoder.resnet import ResNet18


class ResNet18SegNet(nn.Module):
    def __init__(self, num_classes, train_encoder=False, momentum=0.5):
        super(ResNet18SegNet, self).__init__()
        self.encoder = ResNet18(train_cnn=train_encoder)
        self.decoder = SegNet(num_classes, momentum=momentum)
        self.name = "ResNet18SegNet"

    def forward(self, inputs):
        x = self.encoder(inputs)
        x = self.decoder(x)
        return x

    def unfreeze(self):
        self.encoder.unfreeze()


if __name__ == "__main__":
    model = ResNet18SegNet(num_classes=3)
    print(model)
