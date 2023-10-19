import torch
import torch.nn as nn
from model.GuidedFilter import GuidedFilter

class UNet(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.drop4 = nn.Dropout(0.5)
        self.pool4 = nn.MaxPool2d(2)

        self.conv5_1 = nn.Conv2d(512, 1024, 3, padding=1)
        self.conv5_2 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.drop5 = nn.Dropout(0.5)

        # Replace Upsample + Conv2d with ConvTranspose2d for the upsampling steps
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6_2 = nn.Conv2d(1024, 512, 3, padding=1)  # Adjust in_channels to 1024 due to concatenation
        self.conv6_3 = nn.Conv2d(512, 512, 3, padding=1)

        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7_2 = nn.Conv2d(512, 256, 3, padding=1)   # Adjust in_channels to 512 due to concatenation
        self.conv7_3 = nn.Conv2d(256, 256, 3, padding=1)

        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8_2 = nn.Conv2d(256, 128, 3, padding=1)   # Adjust in_channels to 256 due to concatenation
        self.conv8_3 = nn.Conv2d(128, 128, 3, padding=1)

        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9_2 = nn.Conv2d(128, 64, 3, padding=1)    # Adjust in_channels to 128 due to concatenation
        self.conv9_3 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv9_4 = nn.Conv2d(64, 16, 3, padding=1)

        self.conv10 = nn.Conv2d(16, n_classes, 1)

    def forward(self, x):
        conv1 = self.conv1_2(self.conv1_1(x))
        pool1 = self.pool1(conv1)

        conv2 = self.conv2_2(self.conv2_1(pool1))
        pool2 = self.pool2(conv2)

        conv3 = self.conv3_2(self.conv3_1(pool2))
        pool3 = self.pool3(conv3)

        conv4 = self.conv4_2(self.conv4_1(pool3))
        drop4 = self.drop4(conv4)
        pool4 = self.pool4(drop4)

        conv5 = self.conv5_2(self.conv5_1(pool4))
        drop5 = self.drop5(conv5)

        up6 = self.up6(drop5)
        merge6 = torch.cat([drop4, up6], dim=1)
        conv6 = self.conv6_3(self.conv6_2(merge6))

        up7 = self.up7(conv6)
        merge7 = torch.cat([conv3, up7], dim=1)
        conv7 = self.conv7_3(self.conv7_2(merge7))

        up8 = self.up8(conv7)
        merge8 = torch.cat([conv2, up8], dim=1)
        conv8 = self.conv8_3(self.conv8_2(merge8))

        up9 = self.up9(conv8)
        merge9 = torch.cat([conv1, up9], dim=1)
        conv9 = self.conv9_4(self.conv9_3(self.conv9_2(merge9)))

        conv10 = self.conv10(conv9)

        return conv10


import torch
import torch.nn as nn
import torch.nn.functional as F

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