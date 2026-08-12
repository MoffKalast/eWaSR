from collections import OrderedDict

import torch
from torch import nn
import torchvision
from torchvision.models import segmentation
from torchvision.models.resnet import resnet101, resnet50, resnet34, resnet18
from torch.hub import load_state_dict_from_url

from .decoders import *
from .utils import IntermediateLayerGetter

model_list = [
    'wasr_resnet101', 'wasr_resnet101_imu', 'wasr_resnet50', 'wasr_resnet50_imu', 'deeplab', 
    'wasr_resnet18_imu', 'ewasr_resnet18', 'ewasr_resnet18_imu', 'ewasr_resnet34'
]
    
model_urls = {
    'deeplabv3_resnet101_coco': 'https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth'
}

def get_model(model_name, num_classes=3, pretrained=True, **kwargs):

    imu = model_name.endswith('_imu')
    if model_name.startswith('wasr_resnet101'):
        model = wasr_deeplabv2_resnet101(num_classes=num_classes, pretrained=pretrained, imu=imu, **kwargs)
    elif model_name.startswith('wasr_resnet50'):
        model = wasr_deeplabv2_resnet50(num_classes=num_classes, imu=imu)
    elif model_name == 'deeplab':
        model = deeplabv3_resnet101(num_classes=num_classes, pretrained=pretrained)
    elif model_name.startswith('wasr_resnet18'):
        model = wasr_deeplabv2_resnet18(num_classes=num_classes, imu=imu)
    elif model_name.startswith('ewasr'):
        backbone = model_name.split("_")[1].split("_")[0]
        model = ewasr(num_classes = num_classes, imu = imu, backbone=backbone, **kwargs)
    else:
        raise ValueError('Unknown model: %s' % model_name)

    return model

class WaSR(nn.Module):
    """
    Implements WaSR model from
    `"A water-obstacle separation and refinement network for unmanned surface vehicles"
    <https://arxiv.org/abs/2001.01921>`_.
    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the following keys:
            - "out": last feature map of the backbone (2048 features)
            - "aux": feature map used for the auxiliary separation loss (1024 features)
            - "skip1": high-resolution feature map (skip connection) used in FFM (256 features)
            - "skip2": low-resolution feature map (skip connection) used in ARM2 (512 features)
        decoder (nn.Module): a WaSR decoder module. Takes the backbone outputs (with skip connections)
            and returns a dense segmentation prediction for the classes
        classifier_input_features (int, optional): number of input features required by classifier
    """
    def __init__(self, backbone, decoder, imu=False):
        super(WaSR, self).__init__()

        self.imu = imu

        self.backbone = backbone
        self.decoder = decoder
        self.pass_image = getattr(decoder, 'needs_image', False)

    def forward(self, x):

        image = x['image']
        features = self.backbone(image)

        imu_mask = x['imu_mask'].float().unsqueeze(1) if self.imu else None
        aux = features['aux']

        if self.pass_image:
            x = self.decoder(features['out'], features['aux'], features['skip2'], features['skip1'], imu_mask, image)
        else:
            x = self.decoder(features['out'], features['aux'], features['skip2'], features['skip1'], imu_mask)

        # Return segmentation map and aux feature map
        output = OrderedDict([
            ('out', x),
            ('aux', aux)
        ])

        return output


def wasr_deeplabv2_resnet101(num_classes=3, pretrained=False, imu=False, **kwargs):
    # Pretrained ResNet101 backbone
    backbone = resnet101(pretrained=True, replace_stride_with_dilation=[False, True, True])
    return_layers = {
        'layer4': 'out',
        'layer1': 'skip1',
        'layer2': 'skip2',
        'layer3': 'aux'
    }
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    if imu:
        decoder = IMUDecoder(num_classes)
    else:
        decoder = NoIMUDecoder(num_classes)

    model = WaSR(backbone, decoder, imu=imu)

    # Load pretrained DeeplabV3 weights (COCO)
    if pretrained:
        model_url = model_urls['deeplabv3_resnet101_coco']
        state_dict = load_state_dict_from_url(model_url, progress=True)

        # Only load backbone weights, since decoder is entirely different
        keys_to_remove = [key for key in state_dict.keys() if not key.startswith('backbone.')]
        for key in keys_to_remove: del state_dict[key]

        model.load_state_dict(state_dict, strict=False)

    return model


def wasr_deeplabv2_resnet50(num_classes=3, imu=False):

    # Pretrained ResNet101 backbone
    backbone = resnet50(pretrained=True, replace_stride_with_dilation=[False, True, True])
    return_layers = {
        'layer4': 'out',
        'layer1': 'skip1',
        'layer2': 'skip2',
        'layer3': 'aux'
    }
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    if imu:
        decoder = IMUDecoder(num_classes)
    else:
        decoder = NoIMUDecoder(num_classes)

    model = WaSR(backbone, decoder, imu=imu)

    return model


class SegmentationNet(nn.Module):
    """Segmentation net wrapper for SOTA models."""
    def __init__(self, backbone, decoder):
        super(SegmentationNet, self).__init__()

        self.backbone = backbone
        self.decoder = decoder

    def forward(self, x):
        features = self.backbone(x['image'])

        x = self.decoder(features['out'])

        # Return segmentation map and aux feature map
        output = OrderedDict([
            ('out', x),
            ('aux', features['aux'])
        ])

        return output


def deeplabv3_resnet101(num_classes=3, pretrained=True):
    model = segmentation.deeplabv3_resnet101(pretrained=pretrained, aux_loss=False)
    model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))

    backbone = model.backbone
    decoder = model.classifier

    return_layers = {
            'layer4': 'out',
            'layer3': 'aux'
    }
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = SegmentationNet(backbone, decoder)

    return model


def wasr_deeplabv2_resnet18(num_classes=3, imu=True):
    # Pretrained ResNet18 backbone
    backbone = resnet18(pretrained=True)
    return_layers = {
        'layer4': 'out',
        'layer1': 'skip1',
        'layer2': 'skip2',
        'layer3': 'aux'
    }
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    if imu:
        decoder = IMUDecoderSmall(num_classes)
    else:
        raise NotImplementedError("IMU for Resnet18 not supported.")

    model = WaSR(backbone, decoder, imu=imu)

    return model


# timm's ResNet-B weights are key- and shape-identical to torchvision's, so they load directly.
# Fetched by URL rather than via timm.create_model, since the "model.tag" syntax needs timm >= 0.9.
backbone_weight_urls = {
    'resnet18': {
        'a1_in1k': 'https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet18_a1_0-d63eafa0.pth',
        'a2_in1k': 'https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet18_a2_0-b61bd467.pth'
    },
    'resnet34': {
        'a1_in1k': 'https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet34_a1_0-46f8f793.pth',
        'a2_in1k': 'https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet34_a2_0-82d47d71.pth',
        'a3_in1k': 'https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet34_a3_0-a20cabb6.pth',
        'bt_in1k': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet34-43635321.pth',
        'gluon_in1k': 'https://github.com/rwightman/pytorch-pretrained-gluonresnet/releases/download/v0.1/gluon_resnet34_v1b-c6d82d59.pth'
    }
}

def resnet_backbone(name, weights_tag=None):
    ctor = {'resnet18': resnet18, 'resnet34': resnet34}[name]
    if weights_tag is None:
        return ctor(pretrained=True)

    urls = backbone_weight_urls[name]
    if weights_tag not in urls:
        raise ValueError('Unknown %s weights tag: %s. Available: %s' % (name, weights_tag, ', '.join(sorted(urls))))

    backbone = ctor(pretrained=False)
    backbone.load_state_dict(load_state_dict_from_url(urls[weights_tag], progress=True, map_location='cpu'))
    return backbone


def ewasr(num_classes, imu, backbone, **kwargs):

    if backbone in ("resnet18", "resnet34"):
        bb = resnet_backbone(backbone, kwargs.get("backbone_weights"))
        return_layers = {
            'layer4': 'out',
            'layer1': 'skip1',
            'layer2': 'skip2',
            'layer3': 'aux'
        }
        # both are BasicBlock (expansion=1), so stage widths are identical and the decoder is unchanged
        ch = 512
        bb = IntermediateLayerGetter(bb, return_layers=return_layers)
    else:
        raise ValueError(f"Backbone {backbone} is not supported!")

    decoder = EWaSRDecoder(
        num_classes=3,
        ch= ch, #512 if kwargs.get("ch") is None else kwargs["ch"], 
        L=6 if kwargs.get("L") is None else kwargs["L"], 
        imu = imu,
        mixer="CCCCSS" if kwargs.get("mixer") is None else kwargs["mixer"],
        ch_sim=256 if kwargs.get("ch_sim") is None else kwargs["ch_sim"],
        enricher="SS" if kwargs.get("enricher") is None else kwargs["enricher"],
        project=False if kwargs.get("project") is None else kwargs["project"],
        refine=False if kwargs.get("refine") is None else kwargs["refine"]
    )

    model = WaSR(bb, decoder, imu=imu)

    return model