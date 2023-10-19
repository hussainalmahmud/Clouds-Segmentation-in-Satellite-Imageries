import torch
import torch.nn as nn
from model.GuidedFilter import GuidedFilter
import torchvision.models as models
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        
        # Use a pretrained ResNet34 as encoder
        self.resnet = models.resnet34(pretrained=True)

        # Encoder layers
        self.encoder1 = nn.Sequential(self.resnet.conv1, self.resnet.bn1, self.resnet.relu)
        self.encoder2 = self.resnet.layer1
        self.encoder3 = self.resnet.layer2
        self.encoder4 = self.resnet.layer3
        self.encoder5 = self.resnet.layer4

        # Decoder layers
        # Replace Upsample + Conv2d with ConvTranspose2d for the upsampling steps
        # Decoder layers
        self.up6 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.conv6_2 = nn.Conv2d(768, 512, 3, padding=1)
        self.conv6_3 = nn.Conv2d(512, 512, 3, padding=1)

        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7_2 = nn.Conv2d(384, 256, 3, padding=1)
        self.conv7_3 = nn.Conv2d(256, 256, 3, padding=1)

        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8_2 = nn.Conv2d(192, 128, 3, padding=1)
        self.conv8_3 = nn.Conv2d(128, 128, 3, padding=1)

        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9_2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv9_3 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv9_4 = nn.Conv2d(64, 16, 3, padding=1)

        self.conv10 = nn.Conv2d(16, n_classes, 1)

    def forward(self, x):
        # Encoder path
        x1 = self.encoder1(x)
        print("x1.shape,: ",x1.shape)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        x5 = self.encoder5(x4)

        # Decoder path
        up6 = self.up6(x5)
        merge6 = torch.cat([x4, up6], dim=1)
        conv6 = self.conv6_3(self.conv6_2(merge6))

        up7 = self.up7(conv6)
        merge7 = torch.cat([x3, up7], dim=1)
        conv7 = self.conv7_3(self.conv7_2(merge7))

        up8 = self.up8(conv7)
        merge8 = torch.cat([x2, up8], dim=1)
        conv8 = self.conv8_3(self.conv8_2(merge8))

        up9 = self.up9(conv8)
        print(" up9.shape : ",up9.shape)
        up9 = F.max_pool2d(up9, 2)
        merge9 = torch.cat([x1, up9], dim=1)
        conv9 = self.conv9_4(self.conv9_3(self.conv9_2(merge9)))


        conv10 = self.conv10(conv9)

        return conv10




class RefinedUNetLite(nn.Module):
    def __init__(self, in_channels, n_classes, r=60, eps=1e-4, unet_pretrained=False):
        super(RefinedUNetLite, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.unet = UNet(in_channels, n_classes)
        if unet_pretrained:
            checkpoint = torch.load(unet_pretrained)
            self.unet.load_state_dict(checkpoint['model_state_dict'])
            print(f"UNet restored, from {unet_pretrained}")
        
        self.guided_filter = GuidedFilter()

    def forward(self, x):
        # PyTorch uses channels-first format, so we adjust accordingly for RGB channels
        image = x[:, 1:4, :, :]
        
        # Guidance (convert RGB to grayscale)
        # Here, we use a simple average to convert to grayscale, which may differ from the TensorFlow method
        guide = torch.mean(image, dim=1, keepdim=True)
        
        logits = self.unet(x)
        refined_logits = self.guided_filter(guide, logits, r=60, eps=1e-4)
        
        # return logits, refined_logits
        return refined_logits

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# # model = 
# model = UNet(in_channels=3, n_classes=1).to(device)
# # print(model)

# from torchsummary import summary

# # model.eval()
# # Assuming the input channel is 3 and the spatial dimensions are 128x128
# summary(model, input_size=(3, 512, 512))