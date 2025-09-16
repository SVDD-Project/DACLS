import os
import random
import torch
import optuna
from torch.utils.data import Subset, DataLoader

from dcgan_modelV3 import DCMusicSpectroGAN
from trainV4 import train_dcgan, evaluate_discriminator
#from utilsV3 import pt_load_dataset, create_numbered_folder, melspec_dataset
from utilsV3 import *

# Set random seed
seed = 42
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Hyperparametri fissi
BATCH_SIZE = 32
NZ, NC = 100, 1
NUM_EPOCHS = 50  
TRAIN_FRACTION = 0.3   #percentuale di dataset usata per il training

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
print(f"Using device {device}")

# Carica dataset completi (dal CSV, non più .txt)
dataset_root = os.path.expanduser("~/dataset/melspec_pth_2_backup")

full_bonafide_dataset = melspec_dataset(
    root_path=dataset_root,
    meta_file="melspec_pth_meta.csv",
    split="Training",
    lbl="bonafide"
)
full_spoof_dataset = melspec_dataset(
    root_path=dataset_root,
    meta_file="melspec_pth_meta.csv",
    split="Training",
    lbl="spoof"
)


def objective(trial):
    # Campiona subset
    num_bonafide = int(len(full_bonafide_dataset) * TRAIN_FRACTION)
    num_spoof = int(len(full_spoof_dataset) * TRAIN_FRACTION)

    bonafide_subset = Subset(full_bonafide_dataset, random.sample(range(len(full_bonafide_dataset)), num_bonafide))
    spoof_subset = Subset(full_spoof_dataset, random.sample(range(len(full_spoof_dataset)), num_spoof))

    bonafide_dload = DataLoader(bonafide_subset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    spoof_dload = DataLoader(spoof_subset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # Hyperparametri da ottimizzare
    beta1 = trial.suggest_float("beta1", 0.3, 0.7)
    lr_g = trial.suggest_float("lr_g", 1e-5, 5e-4, log=True)
    lr_d = trial.suggest_float("lr_d", 1e-5, 5e-4, log=True)
    lambda_l1 = trial.suggest_float("lambda_l1", 0.0, 15.0)
    lambda_mse = trial.suggest_float("lambda_mse", 0.0, 15.0)
    lambda_fm = trial.suggest_float("lambda_fm", 0.0, 15.0)
    ngf = trial.suggest_categorical("ngf", [64, 128])
    ndf = trial.suggest_categorical("ndf", [64, 128])
    update_d_every = trial.suggest_categorical("update_d_every", [1])
    dropout_p = trial.suggest_float("dropout_p", 0.0, 0.4)
    noise_std = trial.suggest_float("noise_std", 0.0, 0.1)

    # Inizializza modello
    dcgan = DCMusicSpectroGAN(device)
    netG, netD = dcgan.model(
        nz=NZ, ngf=ngf, nc=NC, ndf=ndf,
        dropout_p=dropout_p,
        update_d_every=update_d_every,
        noise_std=noise_std
    )

    # Crea cartella numerata per i risultati
    result_path = create_numbered_folder("OptunaResults13")

    # Train
    train_dcgan(
        device=device,
        nz=NZ,
        lr_g=lr_g,
        lr_d=lr_d,
        beta1=beta1,
        netD=netD,
        netG=netG,
        bonafide_dload=bonafide_dload,
        spoof_dload=spoof_dload,
        num_epochs=NUM_EPOCHS,
        result_path_numbered=result_path,
        lambda_l1=lambda_l1,
        lambda_mse=lambda_mse,
        lambda_fm=lambda_fm,
        update_d_every=update_d_every,
        dropout_p=dropout_p,
        noise_std=noise_std
    )

    # Valutazione su Validation
    val_metrics = evaluate_discriminator(netD, split="Validation", batch_size=BATCH_SIZE, device=device)
    val_accD = val_metrics[0]

    # Log risultati
    with open("optuna_results13.txt", "a") as f:
        f.write(f"Trial {trial.number} | Val_accD: {val_accD:.4f} | "
                f"beta1: {beta1:.4f}, lr_g: {lr_g:.6f}, lr_d: {lr_d:.6f}, "
                f"lambda_l1: {lambda_l1:.2f}, lambda_mse: {lambda_mse:.2f}, lambda_fm: {lambda_fm:.2f}, "
                f"ngf: {ngf}, ndf: {ndf}, dropout_p: {dropout_p:.2f}, update_d_every: {update_d_every}, noise_std: {noise_std:.3f}\n")

    return val_accD


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=30)
    args = parser.parse_args()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.n_trials)

    print("Migliori iperparametri trovati:")
    print(study.best_params)


if __name__ == "__main__":
    main()
