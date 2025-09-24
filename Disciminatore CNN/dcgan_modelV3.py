import torch
import torch.nn as nn

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k_size, stride, dropout_p):
        super(CNNBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, k_size, stride, 1, bias=False, padding_mode="reflect"
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(p=dropout_p),
        )

    def forward(self, x):
        return self.conv(x)


class RefinerGenerator(nn.Module):
    def __init__(self, nz, ngf, nc, noise_std):  # default 0 (no rumore)
        super(RefinerGenerator, self).__init__()
        self.noise_std = noise_std  # valore di deviazione standard del rumore gaussiano
        
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.Conv2d(ngf, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.Conv2d(ngf * 8, ngf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        encoded = self.encoder(input)

        # Aggiunta rumore solo in training se noise_std > 0
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(encoded) * self.noise_std
            encoded = encoded + noise

        decoded = self.decoder(encoded)
        return decoded


class Discriminator(nn.Module):
    """
    Basic CNN for binary classification 
    Input: 1-channel 128x128 maps
    Since no sigmoid and softmax is inserted, use nn.BCEWithLogitsLoss() when training
    """
    def __init__(self, in_channels=1, k_size=(4,4), features=[64, 128, 256, 512], dropout_p=0.0):
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
                CNNBlock(in_channels, feature, k_size, stride=1 if feature == features[-1] else 2, dropout_p=dropout_p),
            )
            in_channels = feature

        layers.extend([
            nn.Conv2d(in_channels, 1, k_size, stride=1, padding=1, padding_mode="reflect"),
            nn.MaxPool2d(kernel_size=2,)]
        )

        self.fc = nn.Linear(7*7, 1)
        self.model = nn.Sequential(*layers)


    def forward(self, x, return_features=False):
        x = self.initial(x)
        features = self.model(x)
        x = features.view(features.size(0), -1)
        x = self.fc(x)
        if return_features:
            return x, features
        return x

            

class DCMusicSpectroGAN():
    def __init__(self, device=None) -> None:
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Custom weights initialization
    def _weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1: #Conv: normale con media 0, std 0.02
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02) #BatchNorm: media 1, std 0.02 e bias a 0, servono a stabilizzare l'addestramento
            nn.init.constant_(m.bias.data, 0)

    def model(self, nz, ngf, nc, ndf, dropout_p, update_d_every, noise_std): #Crea i modelli Generator e Discriminator
        netG = RefinerGenerator(nz, ngf, nc, noise_std).to(self.device)
        netD = Discriminator(in_channels=1, k_size=4, features=[64, 128, 256, 512], dropout_p=dropout_p).to(self.device) #

        # Initialize weights
        netG.apply(self._weights_init)
        netD.apply(self._weights_init)

        return (netG, netD) 
