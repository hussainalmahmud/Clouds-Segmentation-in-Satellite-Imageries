import segmentation_models_pytorch as smp


def pretrained_UNet(encoder, in_channels=3, classes=1):
    aux_params=dict(
        pooling='avg',             # one of 'avg', 'max'
        dropout=0.5,               # dropout ratio, default is None
        activation='sigmoid',      # activation function, default is None
        classes=1,                 # define number of output labels
    )

    model = smp.Unet(encoder, in_channels=in_channels, classes=1, aux_params=aux_params)
    return model

