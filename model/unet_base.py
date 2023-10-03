import yaml
import timm 
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.base.initialization import initialize_decoder
from segmentation_models_pytorch.base import modules as md


cfg = {
    'model': {
        'encoder': 'maxvit_tiny_tf_512.in1k',
        'pretrained': True,
        'decoder_channels': [256, 128, 64, 32, 16],
        'dropout': 0.0,
        'in_channels': 3,
        'n_classes': 1,
    }
}

"""U-Net Model"""
"""
U-Net decoder from Segmentation Models PyTorch
https://github.com/qubvel/segmentation_models.pytorch
"""
class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        use_batchnorm=True,
        dropout=0,
    ):
        super().__init__()

        conv_in_channels = in_channels + skip_channels

        # Convolve input embedding and upscaled embedding
        self.conv1 = md.Conv2dReLU(
            conv_in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

        self.dropout_skip = nn.Dropout(p=dropout)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        if skip is not None:
            skip = self.dropout_skip(skip)
            x = torch.cat([x, skip], dim=1)

        x = self.conv1(x)
        x = self.conv2(x)

        return x


class UnetDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        use_batchnorm=True,
        dropout=0,
    ):
        super().__init__()

        encoder_channels = encoder_channels[::-1]

        # Computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        self.center = nn.Identity()

        # Combine decoder keyword arguments
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, use_batchnorm=use_batchnorm, dropout=dropout)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, features):
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x


def _check_reduction(reduction_factors):
    """
    Assume spatial dimensions of the features decrease by factors of two.
    For example, convnext start with stride=4 cannot be used in my code.
    """
    r_prev = 1
    for r in reduction_factors:
        if r / r_prev != 2:
            raise AssertionError('Reduction assumed to increase by 2: {}'.format(reduction_factors))
        r_prev = r

class UNet_base(nn.Module):
    # The main U-Net model
    # See also TimmUniversalEncoder in Segmentation Models PyTorch
    def __init__(self, pretrained=True, tta=None):
        super().__init__()
        name = cfg['model']['encoder']
        print('Model name:', name)
        dropout = cfg['model']['dropout']
        pretrained = pretrained and cfg['model']['pretrained']
        self.in_channels = cfg['model']['in_channels']
        self.n_classes = cfg['model']['n_classes']
        self.bilinear = False
        self.encoder = timm.create_model(name, 
                                        features_only=True, 
                                        pretrained=pretrained, 
                                         in_chans=cfg['model']['in_channels'],
                                        )
        encoder_channels = self.encoder.feature_info.channels()

        _check_reduction(self.encoder.feature_info.reduction())

        decoder_channels = cfg['model']['decoder_channels']  # (256, 128, 64, 32, 16)
        print('Encoder channels:', name, encoder_channels)
        print('Decoder channels:', decoder_channels)

        assert len(encoder_channels) == len(decoder_channels)

        self.decoder = UnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            dropout=dropout,
        )

        self.segmentation_head = smp.base.SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=1, activation=None, kernel_size=3,
        )


        initialize_decoder(self.decoder)        

    def forward(self, x):
        features = self.encoder(x)
        decoder_output = self.decoder(features)
        y_pred = self.segmentation_head(decoder_output)

        return y_pred