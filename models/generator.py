import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseBlock(nn.Module):
    def __init__(self, in_channels=64, growth_rate=32, res_scale=0.2):
        super().__init__()
        self.res_scale = res_scale

        self.conv1 = nn.Conv2d(in_channels, growth_rate, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_channels + growth_rate, growth_rate, 3, 1, 1)
        self.conv3 = nn.Conv2d(in_channels + 2 * growth_rate, growth_rate, 3, 1, 1)
        self.conv4 = nn.Conv2d(in_channels + 3 * growth_rate, growth_rate, 3, 1, 1)
        self.conv5 = nn.Conv2d(in_channels + 4 * growth_rate, in_channels, 3, 1, 1)
        
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * self.res_scale + x


class RRDB(nn.Module):
    def __init__(self, in_channels=64, growth_rate=32, res_scale=0.2):
        super().__init__()
        self.res_scale = res_scale
        self.rdb1 = DenseBlock(in_channels, growth_rate, res_scale)
        self.rdb2 = DenseBlock(in_channels, growth_rate, res_scale)
        self.rdb3 = DenseBlock(in_channels, growth_rate, res_scale)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * self.res_scale + x


class RRDBNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_features=64, num_blocks=4, growth_rate=32, scale_factor=4):
        super().__init__()
        self.scale_factor = scale_factor

        self.conv_first = nn.Conv2d(in_channels, num_features, 3, 1, 1)
        self.rrdbs = nn.Sequential(*[
            RRDB(num_features, growth_rate) for _ in range(num_blocks)
        ])
        self.conv_trunk = nn.Conv2d(num_features, num_features, 3, 1, 1)

        self.upconv1 = nn.Conv2d(num_features, num_features, 3, 1, 1)
        self.upconv2 = nn.Conv2d(num_features, num_features, 3, 1, 1)

        self.hr_conv = nn.Conv2d(num_features, num_features, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_features, out_channels, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        fea_first = self.conv_first(x)
        trunk = self.rrdbs(fea_first)
        trunk = self.conv_trunk(trunk)
        fea = fea_first + trunk

        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))

        out = self.conv_last(self.lrelu(self.hr_conv(fea)))
        return torch.clamp(out, 0.0, 1.0)


class SRResNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_features=64, num_blocks=4, scale_factor=4):
        super().__init__()
        self.conv_first = nn.Sequential(
            nn.Conv2d(in_channels, num_features, 9, 1, 4),
            nn.PReLU()
        )

        res_blocks = []
        for _ in range(num_blocks):
            res_blocks.append(nn.Sequential(
                nn.Conv2d(num_features, num_features, 3, 1, 1),
                nn.BatchNorm2d(num_features),
                nn.PReLU(),
                nn.Conv2d(num_features, num_features, 3, 1, 1),
                nn.BatchNorm2d(num_features)
            ))
        self.res_blocks = nn.ModuleList(res_blocks)

        self.conv_mid = nn.Sequential(
            nn.Conv2d(num_features, num_features, 3, 1, 1),
            nn.BatchNorm2d(num_features)
        )

        self.upsample = nn.Sequential(
            nn.Conv2d(num_features, num_features * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.PReLU(),
            nn.Conv2d(num_features, num_features * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.PReLU()
        )

        self.conv_last = nn.Conv2d(num_features, out_channels, 9, 1, 4)

    def forward(self, x):
        fea_first = self.conv_first(x)
        fea = fea_first

        for block in self.res_blocks:
            fea = fea + block(fea)

        fea = self.conv_mid(fea) + fea_first
        fea = self.upsample(fea)
        out = self.conv_last(fea)
        return torch.clamp(out, 0.0, 1.0)
