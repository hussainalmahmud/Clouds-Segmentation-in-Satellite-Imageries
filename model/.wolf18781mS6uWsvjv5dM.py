from torch.cuda.amp import autocast
import torch.nn as nn
import segmentation_models_pytorch as smp
class SegModel(nn.Module):
    def __init__(self, encoder: str, network: str, in_channels: int = 3, n_class: int = 1,
                 pre_train="imagenet", **kwargs):
        self.in_channels = in_channels  
        self.n_classes = n_class  
        super(SegModel, self).__init__()
        self.smp_model_name = ["Unet","DeepLabV3Plus"]
        self.model = getattr(smp,network)(
            encoder_name=encoder,encoder_weights=pre_train,in_channels=in_channels,classes=n_class,
            
        )
    @autocast()
    def forward(self, x):
        x = self.model(x)
        return x
