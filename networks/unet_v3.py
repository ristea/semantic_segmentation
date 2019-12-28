from networks.unet_modules import *


class UNetV3(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNetV3, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up_reduce(1024, 512, bilinear)
        self.up2 = Up_reduce(512, 256, bilinear)
        self.up3 = Up_reduce(256, 128, bilinear)
        self.up4 = Up_reduce(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        self.ca = CALayer(21)

        self.out_conv = nn.Conv2d(in_channels=21*5, out_channels=n_classes, kernel_size=3, padding=1)

        self.upsample1 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=16),
            nn.Conv2d(in_channels=1024, out_channels=21, kernel_size=3, padding=1),
        )
        self.upsample2 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=8),
            nn.Conv2d(in_channels=512, out_channels=21, kernel_size=3, padding=1),
        )
        self.upsample3 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=4),
            nn.Conv2d(in_channels=256, out_channels=21, kernel_size=3, padding=1),
        )
        self.upsample4 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels=128, out_channels=21, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x5 = self.down4(x4)
        feat1 = self.upsample1(x5)
        feat1 = self.ca(feat1)

        x = self.up1(x5, x4)
        feat2 = self.upsample2(x)
        feat2 = self.ca(feat2)

        x = self.up2(x, x3)
        feat3 = self.upsample3(x)
        feat3 = self.ca(feat3)

        x = self.up3(x, x2)
        feat4 = self.upsample4(x)
        feat4 = self.ca(feat4)

        x = self.up4(x, x1)
        feat5 = self.outc(x)
        feat5 = self.ca(feat5)

        all_maps = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)
        return self.out_conv(all_maps)


class UNetV3_2(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNetV3_2, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up_reduce(1024, 512, bilinear)
        self.up2 = Up_reduce(512, 256, bilinear)
        self.up3 = Up_reduce(256, 128, bilinear)
        self.up4 = Up_reduce(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        self.se1 = SELayer(0)
        self.se2 = SELayer(0)
        self.se3 = SELayer(0)
        self.se4 = SELayer(0)
        self.se5 = SELayer(0)

        self.out_conv = nn.Conv2d(in_channels=21*5, out_channels=n_classes, kernel_size=3, padding=1)

        self.upsample1 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=16),
            nn.Conv2d(in_channels=1024, out_channels=21, kernel_size=3, padding=1),
        )
        self.upsample2 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=8),
            nn.Conv2d(in_channels=512, out_channels=21, kernel_size=3, padding=1),
        )
        self.upsample3 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=4),
            nn.Conv2d(in_channels=256, out_channels=21, kernel_size=3, padding=1),
        )
        self.upsample4 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels=128, out_channels=21, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x5 = self.down4(x4)
        feat1 = self.upsample1(x5)
        feat1 = self.se1(feat1)

        x = self.up1(x5, x4)
        feat2 = self.upsample2(x)
        feat2 = self.se2(feat2)

        x = self.up2(x, x3)
        feat3 = self.upsample3(x)
        feat3 = self.se3(feat3)

        x = self.up3(x, x2)
        feat4 = self.upsample4(x)
        feat4 = self.se4(feat4)

        x = self.up4(x, x1)
        feat5 = self.outc(x)
        feat5 = self.se5(feat5)

        all_maps = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)
        return self.out_conv(all_maps)