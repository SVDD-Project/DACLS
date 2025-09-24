import torch
import torch.nn as nn

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
    def __init__(self, nc, ndf, dropout_p):
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
        netD = Discriminator(nc, ndf, dropout_p).to(self.device) #

        # Initialize weights
        netG.apply(self._weights_init)
        netD.apply(self._weights_init)

        return (netG, netD) 