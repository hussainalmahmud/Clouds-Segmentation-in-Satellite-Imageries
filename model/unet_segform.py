import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp 
from transformers import SegformerForSemanticSegmentation
from data_utils import *
import torch
import torch.nn as nn
from timm.models.efficientnet import *
import segmentation_models_pytorch as smp
import segmentation_models_pytorch as smp
import pandas as pd
import os

import os
from typing import List
import numpy as np
import pandas as pd
import torch
import torchvision
import torch
import gc
import typer

class SegFormer_b1(nn.Module):
    def __init__(self):
        super(SegFormer_b1, self).__init__()
        self.segformer = SegformerForSemanticSegmentation.from_pretrained('nvidia/segformer-b1-finetuned-ade-512-512')
        self.segformer.decode_head.classifier = nn.Conv2d(256,1,kernel_size=1)
    # @torch.cuda.amp.autocast()
    def forward(self, image):
        image = image[:,0:3]
        
        batch_size = len(image)
        with amp.autocast():
            mask = self.segformer(image).logits
            mask = F.interpolate(mask, image.shape[-2:], mode="bilinear", align_corners=True)
            
        return mask
    

class AmpNet_2(SegFormer_b1):
    
    def __init__(self):
        super(AmpNet, self).__init__()
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
        super(AmpNet, self).__init__(params)
    @torch.cuda.amp.autocast()
    def forward(self,*args):
        return super(AmpNet, self).forward(*args)
    

hparams = {
    "backbone": 'timm-efficientnet-b1',
    "weights": "noisy-student",
}


def getEffv2Model(hparams):
    # unet_model = UnetEffNetV2(hparams)
    unet_model = AmpNet(hparams)
    unet_model.cuda()
    return unet_model

def getEffb1Model4ch(hparams):
    unet_model = SegFormer_b1(hparams)
    unet_model.cuda()
    return unet_model