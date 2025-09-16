import os
import argparse
import numpy as np
import random
import torch

from dcgan_modelV2 import DCMusicSpectroGAN
from trainV4 import train_dcgan

from utilsV3 import *

# Set random seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def argparser():
    parser = argparse.ArgumentParser()
#    parser.add_argument('-gan', "--gan", required=True, type=str, choices=["dcgan", "cgan"],
#                        help="Specify GAN model architecture.")
#    parser.add_argument('-d', "--dataset", required=True, type=str,
#                        help="Path to spectrogram images dataset.")
    parser.add_argument('-log', action="store_true", help="Use log spectrograms, save to Results_log/")
#    parser.add_argument('-mel', action="store_true", help="Use mel spectrograms, save to Results_mel/")
    return parser


def create_numbered_folder(base_path):
    
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    i = 1
    while True:
        new_path = os.path.join(base_path, str(i))
        if not os.path.exists(new_path):
            os.makedirs(new_path)
            return new_path
        i += 1


def main():
    args = argparser().parse_args()

    # Determina la cartella base dei risultati
    if args.log:
        base_results_folder = "Results_log"
    else:
        base_results_folder = "Results_log"

    # Crea cartella numerata all'interno di quella base
    result_path = create_numbered_folder(base_results_folder)

    
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    print(f'Using device {device}')

    # Hyperparameters
    batch_size = 32
    # image_size = (128, 128)
    nc = 1    # number of channels
    nz = 100  # Size of the latent vector
    ngf = 64  # Number of generator filters
    ndf = 64  # Number of discriminator filters
    num_epochs = 100
    #lr = 0.0002
    beta1 = 0.6226

    lambda_l1 = 5.46
    lambda_mse = 6.62
    lambda_fm = 5.68
    #lambda_sc = 0
    #lambda_hinge = 0
    lr_g = 0.000228
    lr_d = 0.000264
    dropout_p = 0.43
    update_d_every = 10
    noise_std = 0.028
    

    # Load the datasets
    bonafide_dload = pt_load_dataset(batch_size, split='Training', lbl='bonafide') # [1 = bonafide]
    spoof_dload = pt_load_dataset(batch_size, split='Training', lbl='spoof') # [0 = spoof]

    dc_msgan = DCMusicSpectroGAN(device)
    netG, netD = dc_msgan.model(nz, ngf, nc, ndf, dropout_p, update_d_every, noise_std)

    # Train
    train_dcgan(
        device=device,
        nz=nz,
        lr_g=lr_g,
        lr_d=lr_d,
        beta1=beta1,
        netD=netD,
        netG=netG,
        spoof_dload=spoof_dload,
        bonafide_dload=bonafide_dload,
        num_epochs=num_epochs,
        result_path_numbered=result_path,
        lambda_l1=lambda_l1,
        lambda_mse=lambda_mse,
        lambda_fm=lambda_fm,
        #lambda_sc=lambda_sc,
        #lambda_hinge=lambda_hinge,
        dropout_p=dropout_p,
        update_d_every=update_d_every,
        noise_std=noise_std
    )

if __name__ == "__main__":
    main()