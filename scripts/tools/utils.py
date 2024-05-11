import os
import toml
import torch
import shutil
import numpy as np
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter


def same_seed(seed):
    """Fixes random number generator seeds for reproducibility."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_config_toml(toml_file_path):
    with open(toml_file_path, "r") as f:
        config = toml.load(f)
    return config


def create_tb_writer(log_dir: str, 
                     accelerator: Accelerator):

    if accelerator.is_main_process:
        tb_writer = SummaryWriter(log_dir)
    else:
        temp_dir = os.path.join(log_dir, "remove")
        tb_writer = SummaryWriter(temp_dir)
        tb_writer.close()
        shutil.rmtree(temp_dir)
    
    return tb_writer


class Accumulator():
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
