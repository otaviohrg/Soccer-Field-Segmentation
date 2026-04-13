import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152

_VARIANTS = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
}


class ResNetEncoder(nn.Module):
    def __init__(self, variant, train_cnn=False):
        super().__init__()
        self.model = _VARIANTS[variant](weights='IMAGENET1K_V1')
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
