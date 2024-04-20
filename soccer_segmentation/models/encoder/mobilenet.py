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

    def forward(self, images):
        results = []
        x = images
        for ii, m in enumerate(self.model.features):
            if ii in {2, 8, 9}:
                for li, l in enumerate(m.block):
                    x = l(x)
                    if li == 0:
                        results.append(x)
            else:
                x = m(x)
                if ii == 12 or ii == 0:
                    results.append(x)
        return results


class MobileNetV3Large(nn.Module):
    def __init__(self, train_cnn=False):
        super(MobileNetV3Large, self).__init__()
        self.model = mobilenet_v3_large(weights='IMAGENET1K_V1')
        if not train_cnn:
            self.freeze()

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, images):
        results = []
        x = images
        for ii, m in enumerate(self.model.features):
            if ii in {5, 7, 11}:
                for li, l in enumerate(m.block):
                    x = l(x)
                    if li == 0:
                        results.append(x)
            else:
                x = m(x)
                if ii == 16 or ii == 0:
                    results.append(x)
        return results


if __name__ == "__main__":
    model = MobileNetV3Large(train_cnn=True)
    print(model.model)
