import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, nc = 1, ndf = 64, dropout_p = 0.43):
        super(Discriminator, self).__init__()
        self.layer0 = nn.Sequential(  # 128x128 -> 64x64
            nn.Conv2d(nc, ndf // 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.layer1 = nn.Sequential(  # 64x64 -> 32x32
            nn.Conv2d(ndf // 2, ndf, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Dropout2d(dropout_p),
        )
        self.layer2 = nn.Sequential(  # 32x32 -> 16x16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Dropout2d(dropout_p),
        )
        self.layer3 = nn.Sequential(  # 16x16 -> 8x8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout_p),
        )
        self.layer4 = nn.Sequential(  # 8x8 -> 4x4
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout_p),
        )
        self.final = nn.Sequential(  # 8x8 -> 1x1
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),        
            nn.Sigmoid()
        )


    def forward(self, input, return_features=False):
        x = self.layer0(input)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        features = x
        out = self.final(features)
        out = out.view(-1)

        if return_features:
            return out, features
        else:
            return out


if __name__ == "__main__":
    x = torch.randn((1, 1, 128, 128))
    model = Discriminator()
    out = model(x)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Learnable parameters: {num_params}')
    print(f'In: {x.shape} - out: {out.shape}')
