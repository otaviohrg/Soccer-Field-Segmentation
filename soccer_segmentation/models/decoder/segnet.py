from typing import List, Optional, Tuple

import torch.nn as nn
import torch.nn.functional as F


def _conv_bn_relu(in_ch: int, out_ch: int, momentum: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch, momentum=momentum),
        nn.ReLU(inplace=True),
    )


def _decode_stage(in_ch: int, out_ch: int, n_convs: int, momentum: float) -> nn.Sequential:
    """n_convs-1 same-channel convs followed by one reducing conv."""
    layers = [_conv_bn_relu(in_ch, in_ch, momentum) for _ in range(n_convs - 1)]
    layers.append(_conv_bn_relu(in_ch, out_ch, momentum))
    return nn.Sequential(*layers)


class SegNetDecoder(nn.Module):
    """
    SegNet-style decoder with configurable channel depths.

    Takes 5 encoder feature maps, re-pools them to obtain max-pool indices,
    then unpools and convolves back to full resolution.

    Args:
        stage_channels: 5 (in_ch, out_ch) tuples, deepest stage first.
                        Stages 0–2 use 3 convolutions each; stage 3 uses 2;
                        stage 4 uses 1 relu conv followed by the output projection.
        out_chn:        Number of output classes.
        momentum:       BatchNorm momentum.
        upsample_input: If set, spatially upsample the encoder feature at this
                        index 2x before pooling. Needed for MobileNet encoders
                        where spatial dimensions do not align with the decoder.
    """

    def __init__(
        self,
        stage_channels: List[Tuple[int, int]],
        out_chn: int = 3,
        momentum: float = 0.5,
        upsample_input: Optional[int] = None,
    ):
        super().__init__()

        self.upsample_input = upsample_input
        if upsample_input is not None:
            self.pre_up = nn.Upsample(scale_factor=2)

        self.pool = nn.MaxPool2d(2, stride=2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2, stride=2)

        n_convs_per_stage = [3, 3, 3, 2]
        self.stages = nn.ModuleList([
            _decode_stage(in_ch, out_ch, n, momentum)
            for (in_ch, out_ch), n in zip(stage_channels[:4], n_convs_per_stage)
        ])

        # Final stage: one relu conv then the output projection (no BN or activation)
        stage4_in, stage4_out = stage_channels[4]
        self.pre_out = _conv_bn_relu(stage4_in, stage4_out, momentum)
        self.out_conv = nn.Conv2d(stage4_out, out_chn, kernel_size=3, padding=1)

    def forward(self, inputs: List):
        x = list(inputs)

        if self.upsample_input is not None:
            x[self.upsample_input] = self.pre_up(x[self.upsample_input])

        pooled, indices, sizes = [], [], []
        for xi in x:
            xi, idx = self.pool(xi)
            pooled.append(xi)
            indices.append(idx)
            sizes.append(xi.size())

        out = pooled[4]
        for i, stage in enumerate(self.stages):
            out = self.unpool(out, indices[4 - i], output_size=sizes[3 - i])
            out = stage(out)

        out = self.unpool(out, indices[0])
        out = self.pre_out(out)
        return self.out_conv(out)
