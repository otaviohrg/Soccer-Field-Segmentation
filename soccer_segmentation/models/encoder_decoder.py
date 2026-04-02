import torch.nn as nn


class EncoderDecoderModel(nn.Module):
    def __init__(self, name, encoder, decoder, small_mask):
        super().__init__()
        self.name = name
        self.encoder = encoder
        self.decoder = decoder
        self.small_mask = small_mask

    def forward(self, inputs):
        x = self.encoder(inputs)
        x = self.decoder(x)
        return x

    def unfreeze(self):
        self.encoder.unfreeze()


class SMPModel(nn.Module):
    def __init__(self, name, model):
        super().__init__()
        self.name = name
        self.small_mask = False
        self.model = model

    def forward(self, inputs):
        return self.model(inputs)

    def unfreeze(self):
        for p in self.model.encoder.parameters():
            p.requires_grad = True
