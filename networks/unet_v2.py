from networks.unet_modules import *


class UNetV2(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNetV2, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        self.out_conv = nn.Conv2d(in_channels=21*5, out_channels=n_classes, kernel_size=3, padding=1)

        self.upsample1 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=16),
            nn.Conv2d(in_channels=512, out_channels=21, kernel_size=3, padding=1),
        )
        self.upsample2 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=8),
            nn.Conv2d(in_channels=256, out_channels=21, kernel_size=3, padding=1),
        )
        self.upsample3 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=4),
            nn.Conv2d(in_channels=128, out_channels=21, kernel_size=3, padding=1),
        )
        self.upsample4 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels=64, out_channels=21, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x5 = self.down4(x4)
        feat1 = self.upsample1(x5)

        x = self.up1(x5, x4)
        feat2 = self.upsample2(x)

        x = self.up2(x, x3)
        feat3 = self.upsample3(x)

        x = self.up3(x, x2)
        feat4 = self.upsample4(x)

        x = self.up4(x, x1)
        feat5 = self.outc(x)

        all_maps = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)
        return self.out_conv(all_maps)