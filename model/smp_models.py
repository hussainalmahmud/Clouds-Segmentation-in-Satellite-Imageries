from torch.cuda.amp import autocast
import torch.nn as nn
import segmentation_models_pytorch as smp
class SegModel(nn.Module):
    def __init__(self, encoder: str, network: str, in_channels: int = 3, n_class: int = 1,
                 pre_train="imagenet", **kwargs):
        self.in_channels = in_channels  
        self.n_classes = n_class  
        
        aux_params=dict(
                pooling='avg',             # one of 'avg', 'max'
                dropout=0.3,               # dropout ratio, default is None
                activation=None,      # activation function, default is None
                classes=1,
            ) 
        super(SegModel, self).__init__()
        self.smp_model_name = ["Unet", "UnetPlusPlus", "MAnet", "Linknet", "FPN", "PSPNet", "DeepLabV3",
                                "DeepLabV3Plus", "PAN"]
        self.model = getattr(smp,network)(
            encoder_name=encoder,encoder_weights=pre_train,in_channels=in_channels,classes=n_class,
            aux_params=aux_params
        )
    @autocast()
    def forward(self, x):
        x = self.model(x)
        return x


# from torch.cuda.amp import autocast
# import torch.nn as nn
# import segmentation_models_pytorch as smp
# import torch
# import torch
# from segmentation_models_pytorch import UnetPlusPlus

# class MultiScaleFeatureLearning(torch.nn.Module):
#     def __init__(self, kernel_sizes):
#         super(MultiScaleFeatureLearning, self).__init__()

#         self.cnns = nn.ModuleList([
#             torch.nn.Conv2d(3, 64, kernel_size, padding=kernel_size // 2) 
#             for kernel_size in kernel_sizes
#         ])

#     def forward(self, x):
#         feature_maps = []
#         for cnn in self.cnns:
#             feature_maps.append(cnn(x))

#         concatenated_feature_map = torch.cat(feature_maps, dim=1)

#         return concatenated_feature_map
# class SegModel(nn.Module):
#     def __init__(self, encoder: str, network: str, in_channels: int = 3, n_class: int = 1,
#                  pre_train="imagenet", **kwargs):
#         super(SegModel, self).__init__()
#         self.in_channels = in_channels  
#         self.n_classes = n_class  
#         self.multi_scale_feature_learning = MultiScaleFeatureLearning([3, 5, 7])
#         # Assuming the output channels of MultiScaleFeatureLearning is 64 * len(kernel_sizes)
#         adjusted_channels = 64 * len([3, 5, 7])
        
#         self.smp_model = getattr(smp, network)(
#             encoder_name=encoder, 
#             encoder_weights=pre_train, 
#             in_channels=adjusted_channels, 
#             classes=n_class
#         )
        
#     @autocast()
#     def forward(self, x):
#         x = self.multi_scale_feature_learning(x)
#         x = self.smp_model(x)
#         return x
