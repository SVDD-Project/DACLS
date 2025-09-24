import os
import torch
import csv
from torch.utils.data import Dataset


def read_meta_file(mf, tgt_split):
    data_buf = []
    with open(mf, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for path, split, label in reader:
            if label == 'spoof':
                label_num = 0
            if label == 'bonafide':
                label_num = 1
            if split == tgt_split:
                data_buf.append((path, label_num))
    return data_buf


class melspec_dataset(Dataset):
    """
    Dataset of torch tensors read from files on disk.
    Return: x (N-dim torch tensor)
            lbl (scalar) 1 = bonafide, 0 = spoof
    """
    def __init__(self, root_path, meta_file, split): # split should be one in ['Training', 'T01', 'T02', 'Validation']
        self.root_path = root_path
        self.meta_file = meta_file
        self.split = split
        self.data_list = read_meta_file(os.path.join(self.root_path, self.meta_file), self.split)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        absolute_path = os.path.join(self.root_path, self.data_list[idx][0])
        x = torch.load(absolute_path)
        # possibly, insert normalization/standardization
        lbl = self.data_list[idx][1]
        return x, lbl


if __name__ == "__main__":
    pass
