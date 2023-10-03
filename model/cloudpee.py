import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.functional as F

smooth = 1e-7

class ASPP(nn.Module):
    def __init__(self, in_channels, out_shape):
        super(ASPP, self).__init__()
        
        # b0
        self.conv1 = nn.Conv2d(in_channels, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.act1 = nn.ReLU()
        
        # b1
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False, dilation=1, groups=in_channels)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.act2 = nn.ReLU()
        # self.pointwise_conv2 = nn.Conv2d(256, 256, 1, bias=False)
        self.pointwise_conv2 = nn.Conv2d(1024, 256, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.act3 = nn.ReLU()

        # b2
        self.conv3 = nn.Conv2d(in_channels, in_channels, 3, padding=3, bias=False, dilation=3, groups=in_channels)
        self.bn4 = nn.BatchNorm2d(in_channels)
        self.act4 = nn.ReLU()
        # self.pointwise_conv3 = nn.Conv2d(256, 256, 1, bias=False)
        self.pointwise_conv3 = nn.Conv2d(1024, 256, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(256)
        self.act5 = nn.ReLU()

        # b3
        self.conv4 = nn.Conv2d(in_channels, in_channels, 3, padding=6, bias=False, dilation=6, groups=in_channels)
        self.bn6 = nn.BatchNorm2d(in_channels)
        self.act6 = nn.ReLU()
        self.pointwise_conv4 = nn.Conv2d(1024, 256, 1, bias=False)
        self.bn7 = nn.BatchNorm2d(256)
        self.act7 = nn.ReLU()

        # b4
        
        self.avg_pool = nn.AdaptiveAvgPool2d((out_shape, out_shape))
        self.pointwise_conv5 = nn.Conv2d(in_channels, 256, 1, bias=False)
        self.bn8 = nn.BatchNorm2d(256)
        self.act8 = nn.ReLU()
        
    def forward(self, x):
        # b0
        x1 = self.act1(self.bn1(self.conv1(x)))

        # b1
        x2 = self.act3(self.bn3(self.pointwise_conv2(self.act2(self.bn2(self.conv2(x))))))

        # b2
        x3 = self.act5(self.bn5(self.pointwise_conv3(self.act4(self.bn4(self.conv3(x))))))

        # b3
        x4 = self.act7(self.bn7(self.pointwise_conv4(self.act6(self.bn6(self.conv4(x))))))

        # b4
        x5 = self.avg_pool(x)
        x5 = self.act8(self.bn8(self.pointwise_conv5(x5)))

        return x  # Modify this return statement according to your final logic, for instance, if you want to concatenate or add layers.

def jacc_coef(y_true, y_pred):
    y_true_f = y_true.view(-1)
    y_pred_f = y_pred.view(-1)
    intersection = (y_true_f * y_pred_f).sum()
    return 1 - ((intersection + smooth) / (y_true_f.sum() + y_pred_f.sum() - intersection + smooth))

class ContrArm(nn.Module):
    def __init__(self, in_channels, filters, kernel_size):
        super(ContrArm, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, filters, kernel_size, padding=kernel_size//2, groups=in_channels, bias=False)

        self.bn1 = nn.BatchNorm2d(filters)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(filters, filters, kernel_size, padding=kernel_size//2, groups=in_channels, bias=False)

        self.bn2 = nn.BatchNorm2d(filters)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels, filters//2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(filters//2)
        self.act3 = nn.ReLU()
        
    def forward(self, x):
        x1 = self.act1(self.bn1(self.conv1(x)))
        x1 = self.act2(self.bn2(self.conv2(x1)))

        x2 = self.act3(self.bn3(self.conv3(x)))
        x = torch.cat([x, x2], dim=1)

        x1 += x
        return F.relu(x1)

class ImprvContrArm(nn.Module):
    def __init__(self, in_channels, filters, kernel_size):
        super(ImprvContrArm, self).__init__()

        # self.conv1 = nn.Conv2d(in_channels, filters, kernel_size, padding=(kernel_size[0]//2, kernel_size[1]//2), groups=in_channels, bias=False)
        self.conv1 = nn.Conv2d(in_channels, filters, kernel_size, padding=kernel_size//2, groups=in_channels, bias=False)

        self.bn1 = nn.BatchNorm2d(filters)
        self.act1 = nn.ReLU()

        # self.conv2 = nn.Conv2d(filters, filters, kernel_size, padding=(kernel_size[0]//2, kernel_size[1]//2), groups=filters, bias=False)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size, padding=kernel_size//2, groups=in_channels, bias=False)

        self.bn2 = nn.BatchNorm2d(filters)
        self.act2 = nn.ReLU()

        # self.conv3 = nn.Conv2d(filters, filters, kernel_size, padding=(kernel_size[0]//2, kernel_size[1]//2), groups=filters, bias=False)
        self.conv3 = nn.Conv2d(filters, filters, kernel_size, padding=kernel_size//2, groups=in_channels, bias=False)

        self.bn3 = nn.BatchNorm2d(filters)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(in_channels, filters//2, (1, 1), bias=False)
        self.bn4 = nn.BatchNorm2d(filters//2)
        self.act4 = nn.ReLU()

        self.conv5 = nn.Conv2d(filters, filters, (1, 1), bias=False)
        self.bn5 = nn.BatchNorm2d(filters)
        self.act5 = nn.ReLU()

    def forward(self, x):
        x0 = self.act1(self.bn1(self.conv1(x)))

        x0_ = self.act2(self.bn2(self.conv2(x0)))

        x1 = self.act3(self.bn3(self.conv3(x0_)))

        x2 = self.act4(self.bn4(self.conv4(x)))

        x2 = torch.cat([x, x2], dim=1)

        x3 = self.act5(self.bn5(self.conv5(x0_)))

        x1 = x1 + x2 + x3
        return F.relu(x1)


class Bridge(nn.Module):
    def __init__(self, in_channels, filters, kernel_size):
        super(Bridge, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, filters, kernel_size, padding=kernel_size//2, groups=in_channels, bias=False)

        self.bn1 = nn.BatchNorm2d(filters)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(filters, filters, kernel_size, padding=kernel_size//2, groups=in_channels, bias=False)

        self.bn2 = nn.BatchNorm2d(filters)
        self.act2 = nn.ReLU()
        self.dropout = nn.Dropout(0.15)

        self.conv3 = nn.Conv2d(in_channels, filters//2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(filters//2)
        self.act3 = nn.ReLU()

    def forward(self, x):
        x0 = self.act1(self.bn1(self.conv1(x)))

        x1 = self.act2(self.bn2(self.conv2(x0)))
        x1 = self.dropout(x1)

        x2 = self.act3(self.bn3(self.conv3(x)))

        x2 = torch.cat([x, x2], dim=1)

        x1 = x1 + x2
        return F.relu(x1)

class ConvBlockExpPath(nn.Module):
    def __init__(self, in_channels, filters, kernel_size):
        super(ConvBlockExpPath, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, filters, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm2d(filters)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(filters, filters, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm2d(filters)
        self.act2 = nn.ReLU()

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        return x


class ConvBlockExpPath3(nn.Module):
    def __init__(self, in_channels, filters, kernel_size):
        super(ConvBlockExpPath3, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, filters, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm2d(filters)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(filters, filters, kernel_size, padding= kernel_size//2)
        self.bn2 = nn.BatchNorm2d(filters)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(filters, filters, kernel_size, padding=kernel_size//2)
        self.bn3 = nn.BatchNorm2d(filters)
        self.act3 = nn.ReLU()

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.act3(self.bn3(self.conv3(x)))
        return x


class AddBlockExpPath(nn.Module):
    def forward(self, x1, x2, x3):
        return F.relu(x1 + x2 + x3)


class ImproveFFBlock4(nn.Module):
    def __init__(self):
        super(ImproveFFBlock4, self).__init__()

        self.pool2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4x4 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.pool8x8 = nn.MaxPool2d(kernel_size=8, stride=8)
        self.pool16x16 = nn.MaxPool2d(kernel_size=16, stride=16)

    def forward(self, x1, x2, x3, x4, pure_ff):
        for _ in range(1):
            x1 = torch.cat([x1, x1], dim=1)
            x1 = self.pool2x2(x1)

        for _ in range(3):
            x2 = torch.cat([x2, x2], dim=1)
        x2 = self.pool4x4(x2)

        for _ in range(7):
            x3 = torch.cat([x3, x3], dim=1)
        x3 = self.pool8x8(x3)

        for _ in range(15):
            x4 = torch.cat([x4, x4], dim=1)
        x4 = self.pool16x16(x4)

        return F.relu(x1 + x2 + x3 + x4 + pure_ff)

class ImproveFFBlock3(nn.Module):
    def __init__(self):
        super(ImproveFFBlock3, self).__init__()

        self.pool2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4x4 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.pool8x8 = nn.MaxPool2d(kernel_size=8, stride=8)

    def forward(self, x1, x2, x3, pure_ff):
        for _ in range(1):
            x1 = torch.cat([x1, x1], dim=1)
        x1 = self.pool2x2(x1)

        for _ in range(3):
            x2 = torch.cat([x2, x2], dim=1)
        x2 = self.pool4x4(x2)

        for _ in range(7):
            x3 = torch.cat([x3, x3], dim=1)
        x3 = self.pool8x8(x3)

        return F.relu(x1 + x2 + x3 + pure_ff)


class ImproveFFBlock2(nn.Module):
    def __init__(self):
        super(ImproveFFBlock2, self).__init__()

        self.pool2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4x4 = nn.MaxPool2d(kernel_size=4, stride=4)

    def forward(self, x1, x2, pure_ff):
        for _ in range(1):
            x1 = torch.cat([x1, x1], dim=1)
        x1 = self.pool2x2(x1)

        for _ in range(3):
            x2 = torch.cat([x2, x2], dim=1)
        x2 = self.pool4x4(x2)

        return F.relu(x1 + x2 + pure_ff)


class ImproveFFBlock1(nn.Module):
    def __init__(self):
        super(ImproveFFBlock1, self).__init__()

        self.pool2x2 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x1, pure_ff):
        for _ in range(1):
            x1 = torch.cat([x1, x1], dim=1)
        x1 = self.pool2x2(x1)

        return F.relu(x1 + pure_ff)



import torch.nn as nn
import torch.nn.functional as F

class ModelArch(nn.Module):
    def __init__(self, input_rows=256, input_cols=256, in_channels=3, n_classes=1):
        super(ModelArch, self).__init__()

        self.in_channels = in_channels
        self.n_classes = n_classes
        # Initial Convolution
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        
        # Contracting path
        self.contr_arm1 = ContrArm(16, 32, 3)
        self.contr_arm2 = ContrArm(32, 64, 3)
        self.contr_arm3 = ContrArm(64, 128, 3)
        self.contr_arm4 = ContrArm(128, 256, 3)
        self.imprv_contr_arm = ImprvContrArm(256, 512, 3)
        
        # Bridge
        self.bridge = Bridge(512, 1024, 3)
        
        # ASPP
        self.aspp = ASPP(1024, 1024/32)  # Assuming input and output channels are same

        # Expansive path
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.upconv5 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)

        self.improve_ff_block4 = ImproveFFBlock4()
        self.improve_ff_block3 = ImproveFFBlock3()
        self.improve_ff_block2 = ImproveFFBlock2()
        self.improve_ff_block1 = ImproveFFBlock1()

        self.conv_block_exp_path3 = ConvBlockExpPath3(1024, 512, 3)
        self.conv_block_exp_path2 = ConvBlockExpPath(768, 256, 3)
        self.conv_block_exp_path1 = ConvBlockExpPath(384, 128, 3)
        self.conv_block_exp_path0 = ConvBlockExpPath(192, 64, 3)
        self.conv_block_exp_path = ConvBlockExpPath(96, 32, 3)
        
        self.add_block_exp_path = AddBlockExpPath()  # Assuming this doesn't change channels

        self.final_conv = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        # Contracting path
        x1 = F.relu(self.conv1(x))
        x2 = self.contr_arm1(x1)
        x2p = F.max_pool2d(x2, 2)

        x3 = self.contr_arm2(x2p)
        x3p = F.max_pool2d(x3, 2)

        x4 = self.contr_arm3(x3p)
        x4p = F.max_pool2d(x4, 2)

        x5 = self.contr_arm4(x4p)
        x5p = F.max_pool2d(x5, 2)

        x6 = self.imprv_contr_arm(x5p)
        x6p = F.max_pool2d(x6, 2)

        x7 = self.bridge(x6p)
        output_shape = x7/32
        x7 = self.aspp(x7)

        # Upsample path
        x8 = self.upconv1(x7)
        ff4 = self.improve_ff_block4(x5, x4, x3, x2)
        x8 = torch.cat([x8, ff4], dim=1)
        x8 = self.conv_block_exp_path3(x8)
        x8 = self.add_block_exp_path(x8, x6, x7)

        x9 = self.upconv2(x8)
        ff3 = self.improve_ff_block3(x4, x3, x2)
        x9 = torch.cat([x9, ff3], dim=1)
        x9 = self.conv_block_exp_path2(x9)
        x9 = self.add_block_exp_path(x9, x5, x8)

        x10 = self.upconv3(x9)
        ff2 = self.improve_ff_block2(x3, x2)
        x10 = torch.cat([x10, ff2], dim=1)
        x10 = self.conv_block_exp_path1(x10)
        x10 = self.add_block_exp_path(x10, x4, x9)

        x11 = self.upconv4(x10)
        ff1 = self.improve_ff_block1(x2)
        x11 = torch.cat([x11, ff1], dim=1)
        x11 = self.conv_block_exp_path0(x11)
        x11 = self.add_block_exp_path(x11, x3, x10)

        x12 = self.upconv5(x11)
        x12 = torch.cat([x12, x1], dim=1)
        x12 = self.conv_block_exp_path(x12)
        x12 = self.add_block_exp_path(x12, x2, x11)

        out = torch.sigmoid(self.final_conv(x12))
        
        return out
