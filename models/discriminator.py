import torch
import torch.nn as nn

class VGGDiscriminator(nn.Module):
    def __init__(self, in_channels=3, num_features=64, img_size=512):
        super().__init__()

        def conv_block(in_f, out_f, stride=1, bn=True):
            layers = [nn.Conv2d(in_f, out_f, kernel_size=3, stride=stride, padding=1)]
            if bn:
                layers.append(nn.BatchNorm2d(out_f))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        layers.extend(conv_block(in_channels, num_features, stride=1, bn=False))
        layers.extend(conv_block(num_features, num_features, stride=2, bn=True))

        layers.extend(conv_block(num_features, num_features * 2, stride=1, bn=True))
        layers.extend(conv_block(num_features * 2, num_features * 2, stride=2, bn=True))

        layers.extend(conv_block(num_features * 2, num_features * 4, stride=1, bn=True))
        layers.extend(conv_block(num_features * 4, num_features * 4, stride=2, bn=True))

        layers.extend(conv_block(num_features * 4, num_features * 8, stride=1, bn=True))
        layers.extend(conv_block(num_features * 8, num_features * 8, stride=2, bn=True))

        layers.extend(conv_block(num_features * 8, num_features * 8, stride=1, bn=True))
        layers.extend(conv_block(num_features * 8, num_features * 8, stride=2, bn=True))

        self.features = nn.Sequential(*layers)

        ds_size = img_size // (2 ** 5)
        self.classifier = nn.Sequential(
            nn.Linear(num_features * 8 * ds_size * ds_size, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        feat = self.features(x)
        feat_flat = torch.flatten(feat, 1)
        out = self.classifier(feat_flat)
        return out
