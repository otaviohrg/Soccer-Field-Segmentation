import segmentation_models_pytorch as smp

from soccer_segmentation.models.decoder.segnet import SegNetDecoder
from soccer_segmentation.models.encoder.mobilenet import MobileNetEncoder
from soccer_segmentation.models.encoder.resnet import ResNetEncoder
from soccer_segmentation.models.encoder.vgg import VGGEncoder
from soccer_segmentation.models.DefaultSegNet import DefaultSegNet
from soccer_segmentation.models.DefaultUNet import DefaultUNet
from soccer_segmentation.models.encoder_decoder import EncoderDecoderModel, SMPModel
from soccer_segmentation.supported_models import supported_encoders, supported_decoders

# SegNet decoder channel configs: 5 (in_ch, out_ch) tuples, deepest stage first.
# Each entry drives the conv widths inside SegNetDecoder.
_VGG_SEGNET          = [(512, 512), (512, 256), (256, 128), (128, 64),    (64, 64)]
_RESNET_SM_SEGNET    = [(512, 256), (256, 128), (128, 64),  (64, 64),     (64, 64)]
_RESNET_LG_SEGNET    = [(2048, 1024), (1024, 512), (512, 256), (256, 64), (64, 64)]
_MOBILENET_SM_SEGNET = [(576, 288), (288, 144), (144, 72),  (72, 16),     (16, 16)]
_MOBILENET_LG_SEGNET = [(960, 480), (480, 240), (240, 120), (120, 16),    (16, 16)]


def _segnet(channels, **kwargs):
    return lambda n: SegNetDecoder(channels, out_chn=n, **kwargs)


_SEGNET_REGISTRY = {
    # (encoder): (encoder_factory, decoder_factory, small_mask)
    'resnet18':        (lambda tc: ResNetEncoder('resnet18',  tc), _segnet(_RESNET_SM_SEGNET),                       True),
    'resnet34':        (lambda tc: ResNetEncoder('resnet34',  tc), _segnet(_RESNET_SM_SEGNET),                       True),
    'resnet50':        (lambda tc: ResNetEncoder('resnet50',  tc), _segnet(_RESNET_LG_SEGNET),                       True),
    'resnet101':       (lambda tc: ResNetEncoder('resnet101', tc), _segnet(_RESNET_LG_SEGNET),                       True),
    'resnet152':       (lambda tc: ResNetEncoder('resnet152', tc), _segnet(_RESNET_LG_SEGNET),                       True),
    'vgg11':           (lambda tc: VGGEncoder('vgg11', tc),        _segnet(_VGG_SEGNET),                             False),
    'vgg13':           (lambda tc: VGGEncoder('vgg13', tc),        _segnet(_VGG_SEGNET),                             False),
    'vgg16':           (lambda tc: VGGEncoder('vgg16', tc),        _segnet(_VGG_SEGNET),                             False),
    'vgg19':           (lambda tc: VGGEncoder('vgg19', tc),        _segnet(_VGG_SEGNET),                             False),
    'mobilenetv3small': (lambda tc: MobileNetEncoder('mobilenetv3small', tc), _segnet(_MOBILENET_SM_SEGNET, upsample_input=2), True),
    'mobilenetv3large': (lambda tc: MobileNetEncoder('mobilenetv3large', tc), _segnet(_MOBILENET_LG_SEGNET, upsample_input=1), True),
}

# Mapping from our encoder names to SMP encoder names
_SMP_ENCODER_NAMES = {
    'resnet18':         'resnet18',
    'resnet34':         'resnet34',
    'resnet50':         'resnet50',
    'resnet101':        'resnet101',
    'resnet152':        'resnet152',
    'vgg11':            'vgg11',
    'vgg13':            'vgg13',
    'vgg16':            'vgg16',
    'vgg19':            'vgg19',
    'mobilenetv3small': 'timm-mobilenetv3_small_100',
    'mobilenetv3large': 'timm-mobilenetv3_large_100',
}


def _create_smp_unet(encoder, num_classes, train_encoder):
    smp_encoder = _SMP_ENCODER_NAMES[encoder]
    model = smp.Unet(
        encoder_name=smp_encoder,
        encoder_weights='imagenet',
        in_channels=3,
        classes=num_classes,
        activation=None,
    )
    if not train_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False
    return SMPModel(name=f"{encoder}unet", model=model)


def create_model(encoder, decoder, num_classes, train_encoder=False):
    assert encoder in supported_encoders, f"Unsupported encoder '{encoder}'. Choose from: {supported_encoders}"
    assert decoder in supported_decoders, f"Unsupported decoder '{decoder}'. Choose from: {supported_decoders}"

    if encoder == 'default':
        if decoder == 'segnet':
            return DefaultSegNet(num_classes=num_classes)
        return DefaultUNet(num_classes=num_classes)

    if decoder == 'unet':
        return _create_smp_unet(encoder, num_classes, train_encoder)

    encoder_factory, decoder_factory, small_mask = _SEGNET_REGISTRY[encoder]
    return EncoderDecoderModel(
        name=f"{encoder}{decoder}",
        encoder=encoder_factory(train_encoder),
        decoder=decoder_factory(num_classes),
        small_mask=small_mask,
    )
