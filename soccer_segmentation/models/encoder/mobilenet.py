import torch.nn as nn
from torchvision.models import mobilenet_v3_small, mobilenet_v3_large

class MobileNetV3Small(nn.Module):
    def __init__(self, train_cnn=False):
        super(MobileNetV3Small, self).__init__()
        self.model = mobilenet_v3_small(weights='IMAGENET1K_V1')
        if not train_cnn:
            self.freeze()

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True

    #def forward(self, images):
    #    x0 = x = self.model.relu(self.model.bn1(self.model.conv1(images)))
    #    x1 = x = self.model.layer1(self.model.maxpool(x))
    #    x2 = x = self.model.layer2(x)
    #    x3 = x = self.model.layer3(x)
    #    x4 = self.model.layer4(x)

    #    return [x0, x1, x2, x3, x4]


class MobileNetV3Large(nn.Module):
    def __init__(self, train_cnn=False):
        super(MobileNetV3Large, self).__init__()
        self.model = mobilenet_v3_small(weights='IMAGENET1K_V1')
        if not train_cnn:
            self.freeze()

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True

    #def forward(self, images):
    #    x0 = x = self.model.relu(self.model.bn1(self.model.conv1(images)))
    #    x1 = x = self.model.layer1(self.model.maxpool(x))
    #    x2 = x = self.model.layer2(x)
    #    x3 = x = self.model.layer3(x)
    #    x4 = self.model.layer4(x)

    #    return [x0, x1, x2, x3, x4]




if __name__ == "__main__":
    model = MobileNetV3Small(train_cnn=True)
    print(model.model)
