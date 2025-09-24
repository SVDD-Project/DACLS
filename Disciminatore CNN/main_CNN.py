"""
Binary classification CNN framework

Carlo Aironi 09/2025
"""

import os, sys
from datetime import datetime
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from cnn_network import CNNClassifier
#from D_GAN import Discriminator as CNNClassifier
from dataset import melspec_dataset


class Trainer:
    def __init__(self, par, net, opt, loss_fn):
        self.par = par # parameters
        self.net = net.to(self.par["DEVICE"])
        self.opt = opt
        self.loss_fn = loss_fn
        self.watch_h = 0.0 # counter
        self.watch_l = 1e6 # counter


    def train(self, train_dataloader, e):
        print('Train...')
        train_loss_buf = 0.0
        train_acc_buf = 0.0
        self.net.train()
        for x, y in tqdm(train_dataloader):
            x = x.to(self.par["DEVICE"])
            y = y.to(self.par["DEVICE"])
            self.opt.zero_grad()
            y_hat = self.net(x)
            y_hat = y_hat.squeeze(1)
            # loss
            train_loss = self.loss_fn(y_hat, y.float())
            train_loss_buf += train_loss.item() # accumulate loss
            train_loss.backward()
            self.opt.step() # minimize loss
            # accuracy
            y_hat = torch.sigmoid(y_hat)
            preds = (y_hat >= 0.5).long() # 0.5 = soglia del classificatore binario
            train_acc_buf += (preds == y).float().mean().item()

        epoch_loss = train_loss_buf/len(train_dataloader)   # calculate loss
        epoch_acc = train_acc_buf/len(train_dataloader)     # calculate accuracy

        return epoch_loss, epoch_acc


    @torch.no_grad()
    def validate(self, val_dataloader, e):
        print('Validate...')
        val_loss_buf = 0.0
        val_acc_buf = 0.0
        self.net.eval()
        for x, y in tqdm(val_dataloader):
            x = x.to(self.par["DEVICE"])
            y = y.to(self.par["DEVICE"])
            y_hat = self.net(x).squeeze(1)
            # loss
            val_loss = self.loss_fn(y_hat, y.float())
            val_loss_buf += val_loss.item()
            # accuracy
            y_hat = torch.sigmoid(y_hat)
            preds = (y_hat >= 0.5).long() # 0.5 = soglia del classificatore binario
            val_acc_buf += (preds == y).float().mean().item()

        epoch_loss = val_loss_buf/len(val_dataloader)   # calculate loss
        epoch_acc = val_acc_buf/len(val_dataloader)     # calculate accuracy

        self.checkpoint_(e, epoch_loss, 'lower')    # save checkpoint for lowest validation loss
        self.checkpoint_(e, epoch_acc, 'higher')    # save checkpoint for highest validation accuracy

        return epoch_loss, epoch_acc


    def checkpoint_(self, e, watch=None, crit=None):
        if crit == 'higher':
            if watch > self.watch_h:
                checkpoint = {"state_dict": self.net.state_dict(),
                              "optimizer": self.opt.state_dict()}
                chkpt_path = os.path.join(self.par["LOG_PATH"], self.par["EXPERIMENT_ID"], "highest_acc_ckpt.pth")
                torch.save(checkpoint, chkpt_path)
                self.watch_h = watch
                if self.par["VERBOSE"] == True:
                    print(f'>> Network checkpoint saved at epoch {e+1} (validation accuracy increased)')
        elif crit == 'lower':
            if watch < self.watch_l:
                checkpoint = {"state_dict": self.net.state_dict(),
                              "optimizer": self.opt.state_dict()}
                chkpt_path = os.path.join(self.par["LOG_PATH"], self.par["EXPERIMENT_ID"], "lowest_loss_ckpt.pth")
                torch.save(checkpoint, chkpt_path)
                self.watch_l = watch
                if self.par["VERBOSE"] == True:
                    print(f'>> Network checkpoint saved at epoch {e+1} (validation loss decreased)')


def start_experiment(par):
    print(f'START EXPERIMENT {par["EXPERIMENT_ID"]}')
    print(f'>> Using device: {par["DEVICE"]}')

    # create log folder structure
    if not os.path.exists(os.path.join(par["LOG_PATH"], par["EXPERIMENT_ID"])):
        os.makedirs(os.path.join(par["LOG_PATH"], par["EXPERIMENT_ID"]))
    else:
        raise Exception(f'Log folder {os.path.join(par["LOG_PATH"], par["EXPERIMENT_ID"])} already exists!')

    # dump results to a csv log file
    log_file = open(os.path.join(par["LOG_PATH"], par["EXPERIMENT_ID"], "results.csv"), mode="w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["epoch", "train_loss", "train_accuracy", "validation_loss", "validation_accuracy"]) # header

    start_tstamp = datetime.now()

    # define network
    net = CNNClassifier().to(par["DEVICE"])

    # define optimizer
    opt = optim.Adam(net.parameters(), lr=par["LEARNING_RATE"], betas=(0.9, 0.999))

    # define loss fn
    loss_fn = nn.BCEWithLogitsLoss() #questo quando usi CNN_network
    #loss_fn = nn.BCELoss() # questo quando usi D_GAN

    # define dataset(s) and dataloader(s)
    train_dataset = melspec_dataset(root_path = par['DATA_PATH'],
                                    meta_file = par['META_FILE'],
                                    split = 'Training')

    # train_dataset = Subset(train_dataset, range(320)) # limit train dataset (only for debug)

    if par["VERBOSE"] == True:
        print(f'>> Loaded {len(train_dataset)} files for Train')

    train_loader = DataLoader(train_dataset,
                              batch_size=par["BATCH_SIZE"],
                              shuffle=True,
                              num_workers=par["NUM_WORKERS"],
                              drop_last=True)

    validation_dataset = melspec_dataset(root_path = par['DATA_PATH'],
                                    meta_file = par['META_FILE'],
                                    split = 'Validation')

    T01_dataset = melspec_dataset(root_path = par['DATA_PATH'],
                                    meta_file = par['META_FILE'],
                                    split = 'T01')  

    T02_dataset = melspec_dataset(root_path = par['DATA_PATH'],
                                    meta_file = par['META_FILE'],
                                    split = 'T02')                                  

    # validation_dataset = Subset(validation_dataset, range(320)) # limit val dataset (only for debug)

    if par["VERBOSE"] == True:
        print(f'>> Loaded {len(validation_dataset)} files for Validation')

    validation_loader = DataLoader(validation_dataset,
                                   batch_size=1,
                                   num_workers=par["NUM_WORKERS"])
    
    T01_loader = DataLoader(T01_dataset,
                            batch_size=1,
                            num_workers=par["NUM_WORKERS"])

    T02_loader = DataLoader(T02_dataset,
                            batch_size=1,
                            num_workers=par["NUM_WORKERS"])

    # Trainer class instance
    T = Trainer(par, net, opt, loss_fn)

    # train/validation routine
    for epoch in range(par["NUM_EPOCHS"]):
        print(f'EPOCH: {epoch+1}/{par["NUM_EPOCHS"]}')
        t_loss, t_acc = T.train(train_loader, epoch)
        v_loss, v_acc = T.validate(validation_loader, epoch)
        log_writer.writerow([epoch+1, t_loss, t_acc, v_loss, v_acc])
        log_file.flush()
        if par["VERBOSE"] == True:
            print(f'EPOCH {epoch+1}/{par["NUM_EPOCHS"]} - Train loss: {t_loss:.3f} - Train acc: {t_acc:.3f} - Validation loss: {v_loss:.3f} - Validation acc: {v_acc:.3f}')
        print("")

    log_file.close()
    end_tstamp = datetime.now()
    if par["VERBOSE"] == True:
        print(f'>> Training started at {start_tstamp.strftime("%m/%d/%Y - %H:%M:%S")}, Training finished at {end_tstamp.strftime("%m/%d/%Y - %H:%M:%S")}')

     # === Carica il checkpoint con la loss più bassa ===
    chkpt_path = os.path.join(par["LOG_PATH"], par["EXPERIMENT_ID"], "lowest_loss_ckpt.pth")
    checkpoint = torch.load(chkpt_path, map_location=par["DEVICE"])
    T.net.load_state_dict(checkpoint["state_dict"])
    if par["VERBOSE"]:
        print(f">> Loaded best checkpoint (lowest validation loss) from {chkpt_path}")

    V_loss, V_acc = T.validate(validation_loader, epoch)
    T01_loss, T01_acc = T.validate(T01_loader, epoch)
    T02_loss, T02_acc = T.validate(T02_loader, epoch)

    # === Salva i risultati in un txt ===
    results_txt_path = os.path.join(par["LOG_PATH"], par["EXPERIMENT_ID"], "final_results.txt")
    with open(results_txt_path, "w") as f:
        f.write("Final evaluation with best model (lowest validation loss)\n")
        f.write(f"Validation -> Loss: {V_loss:.4f}, Accuracy: {V_acc:.4f}\n")
        f.write(f"T01        -> Loss: {T01_loss:.4f}, Accuracy: {T01_acc:.4f}\n")
        f.write(f"T02        -> Loss: {T02_loss:.4f}, Accuracy: {T02_acc:.4f}\n")

    if par["VERBOSE"]:
        print(f">> Final results saved in {results_txt_path}")

if __name__ == "__main__":
    """
    Run $python main.py
    """
    params = {"EXPERIMENT_ID": "002",
              "LOG_PATH": "CNN_Logs",
              "VERBOSE": True,
              "DATA_PATH": "/home/deepfake/dataset/melspec_pth_2_backup",
              "META_FILE": "melspec_pth_meta.csv",
              "NUM_EPOCHS": 100,
              "LEARNING_RATE": 0.0005,
              "BATCH_SIZE": 32,                    # train batch size
              "NUM_WORKERS": 4,
              "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu")
              }

    start_experiment(params)