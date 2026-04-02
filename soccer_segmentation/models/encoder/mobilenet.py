import torch.nn as nn
from torchvision.models import mobilenet_v3_small, mobilenet_v3_large

_VARIANTS = {
    'mobilenetv3small': (mobilenet_v3_small, {2, 8, 9}, 12),
    'mobilenetv3large': (mobilenet_v3_large, {5, 7, 11}, 16),
}


class MobileNetEncoder(nn.Module):
    def __init__(self, variant, train_cnn=False):
        super().__init__()
        model_fn, self._block_indices, self._last_idx = _VARIANTS[variant]
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
            if ii in self._block_indices:
                for li, l in enumerate(m.block):
                    x = l(x)
                    if li == 0:
                        results.append(x)
            else:
                x = m(x)
                if ii == self._last_idx or ii == 0:
                    results.append(x)
        return results
