# full assembly of the sub-parts to form the complete net

import torch.nn.functional as F

from model_parts import *
from torchvision import models

class UNet(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(in_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

def up_conv(in_channels, out_channels):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

class ResNetUNet(nn.Module):
    """Shallow Unet with ResNet18 or ResNet34 encoder."""

    def __init__(self, in_channels, n_classes=12, encoder=models.resnet34):
        super().__init__()
        self.encoder = encoder(pretrained=False)
        self.encoder.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.encoder_layers = list(self.encoder.children())
        
        self.block1 = nn.Sequential(*self.encoder_layers[:3])
        self.block2 = nn.Sequential(*self.encoder_layers[3:5])
        self.block3 = self.encoder_layers[5]
        self.block4 = self.encoder_layers[6]
        self.block5 = self.encoder_layers[7]

        self.up_conv6 = up_conv(512, 512)
        self.conv6 = double_conv(512 + 256, 512)
        self.up_conv7 = up_conv(512, 256)
        self.conv7 = double_conv(256 + 128, 256)
        self.up_conv8 = up_conv(256, 128)
        self.conv8 = double_conv(128 + 64, 128)
        self.up_conv9 = up_conv(128, 64)
        self.conv9 = double_conv(64 + 64, 64)
        self.up_conv10 = up_conv(64, 32)
        self.conv10 = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)

        x = self.up_conv6(block5)
        x = torch.cat([x, block4], dim=1)
        x = self.conv6(x)

        x = self.up_conv7(x)
        x = torch.cat([x, block3], dim=1)
        x = self.conv7(x)

        x = self.up_conv8(x)
        x = torch.cat([x, block2], dim=1)
        x = self.conv8(x)

        x = self.up_conv9(x)
        x = torch.cat([x, block1], dim=1)
        x = self.conv9(x)

        x = self.up_conv10(x)
        x = self.conv10(x)

        return x

class DeepResUnet(nn.Module):
    """Deep Unet with ResNet50, ResNet101 or ResNet152 encoder."""

    def __init__(self, in_channels, n_classes=12, encoder = models.resnet101):
        super().__init__()
        self.encoder = encoder(pretrained=False)
        self.encoder.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.encoder_layers = list(self.encoder.children())

        self.block1 = nn.Sequential(*self.encoder_layers[:3])
        self.block2 = nn.Sequential(*self.encoder_layers[3:5])
        self.block3 = self.encoder_layers[5]
        self.block4 = self.encoder_layers[6]
        self.block5 = self.encoder_layers[7]

        self.up_conv6 = up_conv(2048, 512)
        self.conv6 = double_conv(512 + 1024, 512)
        self.up_conv7 = up_conv(512, 256)
        self.conv7 = double_conv(256 + 512, 256)
        self.up_conv8 = up_conv(256, 128)
        self.conv8 = double_conv(128 + 256, 128)
        self.up_conv9 = up_conv(128, 64)
        self.conv9 = double_conv(64 + 64, 64)
        self.up_conv10 = up_conv(64, 32)
        self.conv10 = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)

        x = self.up_conv6(block5)
        x = torch.cat([x, block4], dim=1)
        x = self.conv6(x)

        x = self.up_conv7(x)
        x = torch.cat([x, block3], dim=1)
        x = self.conv7(x)

        x = self.up_conv8(x)
        x = torch.cat([x, block2], dim=1)
        x = self.conv8(x)

        x = self.up_conv9(x)
        x = torch.cat([x, block1], dim=1)
        x = self.conv9(x)

        x = self.up_conv10(x)
        x = self.conv10(x)

        return x

class VGGUnet(nn.Module):
    """Unet with VGG-16 or VGG-19 encoder.
    """
    def __init__(self, in_channels=1, n_classes=12, encoder=models.vgg19):
        super().__init__()
        self.encoder = encoder(pretrained=False).features
        self.encoder[0] = nn.Conv2d(in_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        if encoder == models.vgg19:
            self.block1 = nn.Sequential(*self.encoder[:4])      # 64
            self.block2 = nn.Sequential(*self.encoder[4:9])     # 128
            self.block3 = nn.Sequential(*self.encoder[9:18])    # 256
            self.block4 = nn.Sequential(*self.encoder[18:27])   # 512
            self.block5 = nn.Sequential(*self.encoder[27:36])   # 512
        elif encoder == models.vgg16:
            self.block1 = nn.Sequential(*self.encoder[:4])      # 64
            self.block2 = nn.Sequential(*self.encoder[4:9])     # 128
            self.block3 = nn.Sequential(*self.encoder[9:16])    # 256
            self.block4 = nn.Sequential(*self.encoder[16:23])   # 512
            self.block5 = nn.Sequential(*self.encoder[23:30])   # 512

        self.up_conv6 = up_conv(512, 512)
        self.conv6 = double_conv(512+512, 512)
        self.up_conv7 = up_conv(512, 256)
        self.conv7 = double_conv(256+256, 256)
        self.up_conv8 = up_conv(256, 128)
        self.conv8 = double_conv(128+128, 128)
        self.up_conv9 = up_conv(128, 64)
        self.conv9 = double_conv(64+64, 64)
        self.conv10 = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)

        x = self.up_conv6(block5)
        x = torch.cat([x, block4], dim=1)
        x = self.conv6(x)

        x = self.up_conv7(x)
        x = torch.cat([x, block3], dim=1)
        x = self.conv7(x)

        x = self.up_conv8(x)
        x = torch.cat([x, block2], dim=1)
        x = self.conv8(x)

        x = self.up_conv9(x)
        x = torch.cat([x, block1], dim=1)
        x = self.conv9(x)

        x = self.conv10(x)
        return x