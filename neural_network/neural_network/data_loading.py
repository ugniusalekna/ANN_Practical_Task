import numpy as np
import os
import torch
from torch.utils.data import Dataset, random_split
import time

class FlowFieldDataset(Dataset):
    def __init__(self, velocity_dir, load_all_data=False):
        start_time = time.time()
        
        self.velocity_dir = velocity_dir
        self.load_all_data = load_all_data

        self.files_true = sorted(os.listdir(os.path.join(self.velocity_dir, 'true_data')))
        self.files_noisy = sorted(os.listdir(os.path.join(self.velocity_dir, 'noisy_data')))

        if self.load_all_data:
            self.data_true = [self.load_one_instance(f, 'true_data') for f in self.files_true]
            self.data_noisy = [self.load_one_instance(f, 'noisy_data') for f in self.files_noisy]

        self.load_time = time.time() - start_time
        print(f'Dataset Loaded in {self.load_time:.4f} s')

    def load_one_instance(self, file_name, data_type):
        u_data = np.load(os.path.join(self.velocity_dir, data_type, file_name))
        return torch.from_numpy(u_data).float()

    def __len__(self):
        return len(self.files_true)

    def __getitem__(self, idx):
        if self.load_all_data:
            u_hr_tensor = self.data_true[idx]
            u_lr_tensor = self.data_noisy[idx]
        else:
            u_hr_tensor = self.load_one_instance(self.files_true[idx], 'true_data')
            u_lr_tensor = self.load_one_instance(self.files_noisy[idx], 'noisy_data')

        u_hr_tensor = u_hr_tensor.permute(2, 0, 1)
        u_lr_tensor = u_lr_tensor.permute(2, 0, 1)

        return u_hr_tensor, u_lr_tensor


def train_test_split(dataset, test_size=0.2):
    total_samples = len(dataset)
    test_sample_size = int(test_size * total_samples)
    train_sample_size = total_samples - test_sample_size

    train_dataset, test_dataset = random_split(dataset, [train_sample_size, test_sample_size])

    return train_dataset, test_dataset