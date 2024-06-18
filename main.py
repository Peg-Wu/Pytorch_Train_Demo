import os
import torch
import warnings
warnings.filterwarnings(action="ignore")
import torchvision as tv
from torch import nn
from trainer import train, test
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader

def get_mnist_datasets(save_dir):
    r"""(C, H, W) --> (1, 28, 28)"""
    os.makedirs(save_dir, exist_ok=True)
    transform = tv.transforms.Compose([tv.transforms.ToTensor(), tv.transforms.Normalize((0.1307), (0.3081))])
    train_ds = tv.datasets.MNIST(root=save_dir, train=True, transform=transform, download=True)
    valid_ds = tv.datasets.MNIST(root=save_dir, train=False, transform=transform, download=True)
    return train_ds, valid_ds


def collate_flatten_mnist(batch_list):
    r"""(N, 1, 28, 28) --> (N, 1*28*28)"""
    batch_size = len(batch_list)
    X = torch.stack([item[0] for item in batch_list], dim=0).reshape(batch_size, -1)
    y = torch.LongTensor([item[1] for item in batch_list])
    return X, y

class Mnist_Model_MLP(nn.Module):
    def __init__(self, mlp_hidden: list = [784, 256, 128, 10]):
        super().__init__()
        assert mlp_hidden[0] == 784 and mlp_hidden[-1] == 10

        layers = []
        for item in zip(mlp_hidden[:-1], mlp_hidden[1:]):
            layers.extend([nn.Linear(*item), nn.ReLU()])

        # remove the last relu activation function
        layers = layers[:-1]
        self.mlp = nn.Sequential(*nn.ModuleList(layers))

    def forward(self, x):
        return self.mlp(x)

# accelerator
accelerator = Accelerator()
device = accelerator.device

# fix random seed
set_seed(520)

# datasets (train_ds: 60000, valid_ds: 10000)
train_ds, valid_ds = get_mnist_datasets("./mnist")

# dataloader
train_dl = DataLoader(train_ds, 
                      shuffle=True,
                      collate_fn=collate_flatten_mnist,
                      batch_size=128)

valid_dl = DataLoader(valid_ds,
                      shuffle=False,
                      collate_fn=collate_flatten_mnist,
                      batch_size=128)

# model
model = Mnist_Model_MLP()

# loss & optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# prepare
model, optimizer, train_dl, valid_dl = accelerator.prepare(model, optimizer, train_dl, valid_dl)

# train
train(epochs = 2,
      model = model,
      train_dl = train_dl,
      valid_dl = valid_dl,
      loss_fn = loss_fn,
      optimizer = optimizer,
      accelerator = accelerator,
      valid_strategy = "epoch",
      valid_interval = 1,
      print_train_logs_strategy = "epoch",
      print_train_logs_interval = 1,
      save_state_global_step = None,
      resume_dir = None,
      early_stop = 2,
      tb_log_dir = "./logs")

# load model
pretrained_model = Mnist_Model_MLP()
pretrained_model.load_state_dict(torch.load("./checkpoints/training_end_model/pytorch_model.bin"))

# test dataloader
test_dl = valid_dl

# prepare
pretrained_model, test_dl = accelerator.prepare(pretrained_model, test_dl)

# test
test(model = pretrained_model, 
     test_dl = test_dl, 
     loss_fn = loss_fn, 
     accelerator = accelerator)