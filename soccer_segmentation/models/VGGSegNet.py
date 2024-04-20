import torch.nn as nn
from soccer_segmentation.models.decoder.segnet_v1 import SegNet
from soccer_segmentation.models.encoder.vgg import VGG11, VGG13, VGG16, VGG19


class VGG16SegNet(nn.Module):
    def __init__(self, num_classes, train_encoder=False, momentum=0.5):
        super(VGG16SegNet, self).__init__()
        self.encoder = VGG16(train_cnn=train_encoder)
        self.decoder = SegNet(num_classes, momentum=momentum)
        self.name = "VGG16SegNet_2"

    def forward(self, inputs):
        x = self.encoder(inputs)
        x = self.decoder(x)
        return x

    def unfreeze(self):
        self.encoder.unfreeze()


if __name__ == "__main__":
    model = VGG16SegNet(num_classes=3)
    print(model)
