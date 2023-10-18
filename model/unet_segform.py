import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp 
from transformers import SegformerForSemanticSegmentation
import torch
import torch.nn as nn

class SegFormer_b1(nn.Module):
    def __init__(self, in_channels=3, n_classes=1):  # default to 3 channels
        super(SegFormer_b1, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        # Assuming Segformer can be initialized with a custom number of channels (it usually can't)
        self.segformer = SegformerForSemanticSegmentation.from_pretrained('nvidia/segformer-b1-finetuned-ade-512-512')  
        
        self.segformer.decode_head.classifier = nn.Conv2d(256, 1, kernel_size=1)

    def forward(self, image):
        image = image[:, :self.in_channels]
        
        batch_size = len(image)
        with amp.autocast():
            mask = self.segformer(image).logits
            mask = F.interpolate(mask, image.shape[-2:], mode="bilinear", align_corners=True)
            
        return mask

    

class AmpNet(SegFormer_b1):
    def __init__(self, in_channels=3, n_classes=1):  # default to 3 channels
        super(AmpNet, self).__init__(in_channels=in_channels, n_classes=n_classes)  # pass in_channels to parent

    @torch.cuda.amp.autocast()
    def forward(self,*args):
        return super(AmpNet, self).forward(*args)


  #True #False






class Net(nn.Module):
    def __init__(self,params):
        super(Net, self).__init__()

        aux_params=dict(
                        pooling='avg',             # one of 'avg', 'max'
                        dropout=0.3,               # dropout ratio, default is None
                        activation=None,      # activation function, default is None
                        classes=1,
                    ) 
        self.unet = smp.Unet(
                    encoder_name=params['backbone'],        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                    encoder_weights=params['weights'],     # use `imagenet` pre-trained weights for encoder initialization
                    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                    decoder_attention_type= None,                      # model output channels (number of classes in your dataset)
                    classes=1,aux_params=aux_params
                    )



    # @torch.cuda.amp.autocast()
    def forward(self, image):
        batch_size = len(image)
        mask,logit = self.unet(image)
        return mask


# In[12]:


import torch.cuda.amp as amp
class AmpNet_1(Net):
    
    def __init__(self,params):
        super(AmpNet_1, self).__init__(params)
    @torch.cuda.amp.autocast()
    def forward(self,*args):
        return super(AmpNet_1, self).forward(*args)
    

hparams = {
    "backbone": 'timm-efficientnet-b1',
    "weights": "noisy-student",
}


# def getEffv2Model(hparams):
#     # unet_model = UnetEffNetV2(hparams)
#     unet_model = AmpNet(hparams)
#     unet_model.cuda()
#     return unet_model

# def getEffb1Model4ch(hparams):
#     unet_model = SegFormer_b1(hparams)
#     unet_model.cuda()
#     return unet_model