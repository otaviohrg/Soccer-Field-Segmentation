import torch.nn as nn
from torchvision.models import vgg11, vgg13, vgg16, vgg19


class VGG11(nn.Module):
    def __init__(self, train_cnn=False):
        super(VGG11, self).__init__()
        self.model = vgg11(weights='IMAGENET1K_V1')
        if not train_cnn:
            self.freeze()

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, images):
        results = []
        x = images
        for ii, m in enumerate(self.model.features):
            x = m(x)
            if ii in {1, 4, 9, 14, 19}:
                results.append(x)
        return results


class VGG13(nn.Module):
    def __init__(self, train_cnn=False):
        super(VGG13, self).__init__()
        self.model = vgg13(weights='IMAGENET1K_V1')
        if not train_cnn:
            self.freeze()

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, images):
        results = []
        x = images
        for ii, m in enumerate(self.model.features):
            x = m(x)
            if ii in {3, 8, 13, 18, 23}:
                results.append(x)
        return results


class VGG16(nn.Module):
    def __init__(self, train_cnn=False):
        super(VGG16, self).__init__()
        self.model = vgg16(weights='IMAGENET1K_V1')
        if not train_cnn:
            self.freeze()

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, images):
        results = []
        x = images
        for ii, m in enumerate(self.model.features):
            x = m(x)
            if ii in {3, 8, 15, 22, 29}:
                results.append(x)
        return results


class VGG19(nn.Module):
    def __init__(self, train_cnn=False):
        super(VGG19, self).__init__()
        self.model = vgg19(weights='IMAGENET1K_V1')
        if not train_cnn:
            self.freeze()

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, images):
        results = []
        x = images
        for ii, m in enumerate(self.model.features):
            x = m(x)
            if ii in {3, 8, 17, 26, 35}:
                results.append(x)
        return results


if __name__ == "__main__":
    model = VGG11(train_cnn=True)
    print(model.model)
