import torch
import torch.nn as nn

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k_size, stride):
        super(CNNBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, k_size, stride, 1, bias=False, padding_mode="reflect"
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv(x)


class CNNClassifier(nn.Module):
    """
    Basic CNN for binary classification 
    Input: 1-channel 128x128 maps
    Since no sigmoid and softmax is inserted, use nn.BCEWithLogitsLoss() when training
    """
    def __init__(self, in_channels=1, k_size=(4,4), features=[64, 128, 256, 512]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels,
                features[0],
                kernel_size=k_size,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2),
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                CNNBlock(in_channels, feature, k_size, stride=1 if feature == features[-1] else 2),
            )
            in_channels = feature

        layers.extend([
            nn.Conv2d(in_channels, 1, k_size, stride=1, padding=1, padding_mode="reflect"),
            nn.MaxPool2d(kernel_size=2,)]
        )

        self.fc = nn.Linear(7*7, 1)
        self.model = nn.Sequential(*layers)


    def forward(self, x):
        x = self.initial(x)
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x # -> [Batch_size, 1]


if __name__ == "__main__":
    x = torch.randn((1, 1, 128, 128))
    model = CNNClassifier()
    out = model(x)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Learnable parameters: {num_params}')
    print(f'In: {x.shape} - out: {out.shape}')
