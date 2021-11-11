import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """
    残差块
    """
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1),
            nn.GroupNorm(out_ch // 16, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            nn.GroupNorm(out_ch // 16, out_ch),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride),
                nn.GroupNorm(out_ch // 16, out_ch),
            )

    def forward(self, x):
        out = self.double_conv(x)
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class ResEncoder(nn.Module):
    """
    7*7conv + 2 resblock + 3*2 resblock
    """
    def __init__(self, in_channel):
        super().__init__()
        self.in_ch = 32
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, 32, kernel_size=7, stride=2, padding=3),
            nn.GroupNorm(32 // 16, 32),
            nn.ReLU()
        )
        self.layer1 = self.make_layer(ResBlock, 32, 2, stride=1)
        self.layer2 = self.make_layer(ResBlock, 64, 2, stride=2)
        self.layer3 = self.make_layer(ResBlock, 128, 2, stride=2)
        self.layer4 = self.make_layer(ResBlock, 256, 2, stride=2)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_ch, channels, stride))
            self.in_ch = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = F.adaptive_avg_pool2d(out, (1, 1))
        return out


class UpBlock(nn.Module):
    """
    upsampling +　doubleconv
    """

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.GroupNorm(out_channels // 16, out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.GroupNorm(out_channels // 16, out_channels),
            nn.ReLU(inplace=True)
        )

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = self.double_conv
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=(2, 2), stride=(2, 2))
            self.conv = self.double_conv

    def forward(self, x):
        x = self.up(x)

        return self.conv(x)


class StressDecoder(nn.Module):
    def __init__(self, out_channel):
        super().__init__()
        self.up1 = UpBlock(256, 128)
        self.up2 = UpBlock(128, 64)
        self.up3 = UpBlock(64, 32)
        self.up4 = UpBlock(32, 16)
        self.outlayer = nn.Conv2d(16, out_channel, kernel_size=(1, 1))

    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.outlayer(x)
        return x


class ResStressnet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.encoder = ResEncoder(in_ch)
        self.decoder = StressDecoder(out_ch)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


if __name__ == '__main__':
    net = ResStressnet(3, 1)
    print(net)
    ipt = torch.randn((1, 3, 224, 224))
    opt = net(ipt)
    print(opt.size())
