import torch.nn as nn
from torchvision.models import vgg16


class VGG16(nn.Module):
    def __init__(self, train_cnn=False):
        super(VGG16, self).__init__()
        self.model = vgg16(weights='IMAGENET1K_V1')
        self.train_cnn = train_cnn

    def unfreeze(self):
        self.train_cnn = True

    def forward(self, images):
        results = []
        x = images
        for ii, m in enumerate(self.model.features):
            x = m(x)
            if ii in {3, 8, 15, 22, 29}:
                results.append(x)
        return results


if __name__ == "__main__":
    model = VGG16(train_cnn=True)
    print(model.model)
