import os
from torch.utils.data import DataLoader, Dataset
import csv
import torch

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


class melspec_dataset(Dataset):
    """
    Dataset of torch tensors read from files on disk.
    Return: x (N-dim torch tensor)
            lbl (scalar) 1 = bonafide, 0 = spoof
    """
    def __init__(self, root_path, meta_file, split, lbl): # split should be one in ['Training', 'T01', 'T02', 'Validation'], lbl should be in [bonafide, spoof]
        self.root_path = root_path
        self.meta_file = meta_file
        self.split = split
        self.lbl = lbl
        self.data_list = read_meta_file(os.path.join(self.root_path, self.meta_file), self.split, self.lbl)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        absolute_path = os.path.join(self.root_path, self.data_list[idx][0])
        x = torch.load(absolute_path, weights_only=True)
        # possibly, insert normalization/standardization
        # Calcolo log-mel con epsilon per evitare log(0)
        
        #passare a log-mel
        epsilon = 1e-6
        x = torch.log10(x + epsilon)
        
        if self.data_list[idx][1] == 'bonafide':
            lbl_num = 1
        if self.data_list[idx][1] == 'spoof':
            lbl_num = 0
        return x, lbl_num

def read_meta_file(mf, tgt_split, tgt_lbl):
    data_buf = []
    with open(mf, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for path, split, label in reader:
            if split == tgt_split and label == tgt_lbl:
                data_buf.append((path, label))
    return data_buf


def pt_load_dataset(batch_size, split, lbl):

    dset = melspec_dataset(root_path = '/home/deepfake/dataset/melspec_pth_2_backup/',
                           meta_file = 'melspec_pth_meta.csv',
                           split = split,
                           lbl = lbl)
    dloader = DataLoader(dset, batch_size=batch_size, shuffle=True, drop_last=True)

    return dloader


# modifiche 


def spectral_convergence_loss(real, fake):
    return torch.norm(real - fake, p='fro') / (torch.norm(real, p='fro') + 1e-8)


def hinge_loss_discriminator(D_real, D_fake):
    loss_real = torch.mean(torch.relu(1.0 - D_real))
    loss_fake = torch.mean(torch.relu(1.0 + D_fake))
    return loss_real + loss_fake

def hinge_loss_generator(D_fake):
    return -torch.mean(D_fake)
