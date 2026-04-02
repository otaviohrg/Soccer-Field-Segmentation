import torch.nn as nn
from torchvision.models import vgg11, vgg13, vgg16, vgg19

_VARIANTS = {
    'vgg11': (vgg11, {1, 4, 9, 14, 19}),
    'vgg13': (vgg13, {3, 8, 13, 18, 23}),
    'vgg16': (vgg16, {3, 8, 15, 22, 29}),
    'vgg19': (vgg19, {3, 8, 17, 26, 35}),
}


class VGGEncoder(nn.Module):
    def __init__(self, variant, train_cnn=False):
        super().__init__()
        model_fn, self._hook_indices = _VARIANTS[variant]
        self.model = model_fn(weights='IMAGENET1K_V1')
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
            x = m(x)
            if ii in self._hook_indices:
                results.append(x)
        return results
