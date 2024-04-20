import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152


class ResNet18(nn.Module):
    def __init__(self, train_cnn=False):
        super(ResNet18, self).__init__()
        self.model = resnet18(weights='IMAGENET1K_V1')
        if not train_cnn:
            self.freeze()

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, images):
        x0 = x = self.model.relu(self.model.bn1(self.model.conv1(images)))
        x1 = x = self.model.layer1(self.model.maxpool(x))
        x2 = x = self.model.layer2(x)
        x3 = x = self.model.layer3(x)
        x4 = self.model.layer4(x)

        return [x0, x1, x2, x3, x4]


class ResNet34(nn.Module):
    def __init__(self, train_cnn=False):
        super(ResNet34, self).__init__()
        self.model = resnet34(weights='IMAGENET1K_V1')
        if not train_cnn:
            self.freeze()

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, images):
        x0 = x = self.model.relu(self.model.bn1(self.model.conv1(images)))
        x1 = x = self.model.layer1(self.model.maxpool(x))
        x2 = x = self.model.layer2(x)
        x3 = x = self.model.layer3(x)
        x4 = self.model.layer4(x)

        return [x0, x1, x2, x3, x4]


class ResNet50(nn.Module):
    def __init__(self, train_cnn=False):
        super(ResNet50, self).__init__()
        self.model = resnet50(weights='IMAGENET1K_V1')
        if not train_cnn:
            self.freeze()

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, images):
        x0 = x = self.model.relu(self.model.bn1(self.model.conv1(images)))
        x1 = x = self.model.layer1(self.model.maxpool(x))
        x2 = x = self.model.layer2(x)
        x3 = x = self.model.layer3(x)
        x4 = self.model.layer4(x)

        return [x0, x1, x2, x3, x4]


class ResNet101(nn.Module):
    def __init__(self, train_cnn=False):
        super(ResNet101, self).__init__()
        self.model = resnet101(weights='IMAGENET1K_V1')
        if not train_cnn:
            self.freeze()

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, images):
        x0 = x = self.model.relu(self.model.bn1(self.model.conv1(images)))
        x1 = x = self.model.layer1(self.model.maxpool(x))
        x2 = x = self.model.layer2(x)
        x3 = x = self.model.layer3(x)
        x4 = self.model.layer4(x)

        return [x0, x1, x2, x3, x4]


class ResNet152(nn.Module):
    def __init__(self, train_cnn=False):
        super(ResNet152, self).__init__()
        self.model = resnet152(weights='IMAGENET1K_V1')
        if not train_cnn:
            self.freeze()

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, images):
        x0 = x = self.model.relu(self.model.bn1(self.model.conv1(images)))
        x1 = x = self.model.layer1(self.model.maxpool(x))
        x2 = x = self.model.layer2(x)
        x3 = x = self.model.layer3(x)
        x4 = self.model.layer4(x)

        return [x0, x1, x2, x3, x4]


if __name__ == "__main__":
    model = ResNet18(train_cnn=True)
    print(model.model)
