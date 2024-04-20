from soccer_segmentation.models.VGGSegNet import *
from soccer_segmentation.models.ResNetSegNet import *
from soccer_segmentation.models.VGGUnet import *
from soccer_segmentation.models.ResNetUnet import *
from soccer_segmentation.models.MobileNetSegNet import *
from soccer_segmentation.models.MobileNetUnet import *
from soccer_segmentation.supported_models import supported_encoders, supported_decoders


def create_model(encoder, decoder, num_classes, train_encoder=False):
    assert encoder in supported_encoders, "Encoder not supported"
    assert decoder in supported_decoders, "Decoder not supported"

    model_name = encoder + decoder
    model = None

    if model_name == 'resnet18segnet':
        model = ResNet18SegNet(num_classes=num_classes, train_encoder=train_encoder)
    elif model_name == 'resnet34segnet':
        model = ResNet34SegNet(num_classes=num_classes, train_encoder=train_encoder)
    elif model_name == 'resnet50segnet':
        model = ResNet50SegNet(num_classes=num_classes, train_encoder=train_encoder)
    elif model_name == 'resnet101segnet':
        model = ResNet101SegNet(num_classes=num_classes, train_encoder=train_encoder)
    elif model_name == 'resnet152segnet':
        model = ResNet152SegNet(num_classes=num_classes, train_encoder=train_encoder)
    elif model_name == 'vgg11segnet':
        model = VGG11SegNet(num_classes=num_classes, train_encoder=train_encoder)
    elif model_name == 'vgg13segnet':
        model = VGG13SegNet(num_classes=num_classes, train_encoder=train_encoder)
    elif model_name == 'vgg16segnet':
        model = VGG16SegNet(num_classes=num_classes, train_encoder=train_encoder)
    elif model_name == 'vgg19segnet':
        model = VGG19SegNet(num_classes=num_classes, train_encoder=train_encoder)
    elif model_name == 'mobilenetv3smallsegnet':
        model = MobileNetV3SmallSegNet(num_classes=num_classes, train_encoder=train_encoder)
    elif model_name == 'mobilenetv3largesegnet':
        model = MobileNetV3LargeSegNet(num_classes=num_classes, train_encoder=train_encoder)
    elif model_name == 'vgg11unet':
        model = VGG11UNet(num_classes=num_classes, train_encoder=train_encoder)
    elif model_name == 'vgg13unet':
        model = VGG13UNet(num_classes=num_classes, train_encoder=train_encoder)
    elif model_name == 'vgg16unet':
        model = VGG16UNet(num_classes=num_classes, train_encoder=train_encoder)
    elif model_name == 'vgg19unet':
        model = VGG19UNet(num_classes=num_classes, train_encoder=train_encoder)
    elif model_name == 'resnet18unet':
        model = ResNet18UNet(num_classes=num_classes, train_encoder=train_encoder)
    elif model_name == 'resnet34unet':
        model = ResNet34UNet(num_classes=num_classes, train_encoder=train_encoder)
    elif model_name == 'resnet50unet':
        model = ResNet50UNet(num_classes=num_classes, train_encoder=train_encoder)
    elif model_name == 'resnet101unet':
        model = ResNet101UNet(num_classes=num_classes, train_encoder=train_encoder)
    elif model_name == 'resnet152unet':
        model = ResNet152UNet(num_classes=num_classes, train_encoder=train_encoder)
    elif model_name == 'mobilenetv3smallsegnet':
        model = MobileNetV3SmallSegNet(num_classes=num_classes, train_encoder=train_encoder)
    elif model_name == 'mobilenetv3largesegnet':
        model = MobileNetV3SmallSegNet(num_classes=num_classes, train_encoder=train_encoder)
    elif model_name == 'mobilenetv3smallunet':
        model = MobileNetV3SmallUNet(num_classes=num_classes, train_encoder=train_encoder)
    elif model_name == 'mobilenetv3largeunet':
        model = MobileNetV3LargeUNet(num_classes=num_classes, train_encoder=train_encoder)

    else:
        raise Exception

    return model
