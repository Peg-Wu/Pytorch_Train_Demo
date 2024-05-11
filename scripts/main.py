import torch
import argparse
import warnings
warnings.filterwarnings(action="ignore")

from tools import *
from torch import nn
from accelerate import Accelerator
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model, PeftModel
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument("--config", "-c", type=str, default="../configs/config.toml", help="Config")
args = parser.parse_args()

# config
config = get_config_toml(args.config)
dc, mc, tc = config["data"], config["model"], config["train"]

# fix random seed
same_seed(config["seed"])

# accelerator
accelerator = Accelerator()
device = accelerator.device

# datasets (train_ds: 60000, valid_ds: 10000)
train_ds, valid_ds = get_mnist_datasets(dc["save_dir"])

# dataloader (取消shuffle参数，否则断点续传会出问题)
train_dl = DataLoader(train_ds, 
                  #     shuffle=True, 
                      collate_fn=collate_flatten_mnist,
                      batch_size=tc["train_batch_size"])

valid_dl = DataLoader(valid_ds, 
                  #     shuffle=False, 
                      collate_fn=collate_flatten_mnist,
                      batch_size=tc["valid_batch_size"])

# model
model = Mnist_Model_MLP(mc["mlp_hidden"])

# add lora adaptor to all linear layers
target_modules = []
for name, param in model.named_parameters():
      target_modules.append('.'.join(name.split(".")[:2]))
target_modules = list(set(target_modules))
lora_config = LoraConfig(target_modules=target_modules)
peft_model = get_peft_model(model, lora_config)

# loss & optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=tc["lr"])

# prepare
peft_model, optimizer, train_dl, valid_dl = accelerator.prepare(peft_model, optimizer, train_dl, valid_dl)

# tensorboard writer
tb_writer = create_tb_writer(log_dir=tc["tb_log_dir"], accelerator=accelerator)

# train
train(epochs=tc["epochs"],
      model=peft_model,
      is_peft_model=tc["is_peft_model"],
      train_dl=train_dl,
      valid_dl=valid_dl,
      loss_fn=loss_fn,
      optimizer=optimizer,
      accelerator=accelerator,
      print_logs_global_steps=tc["print_logs_global_steps"],
      save_state_global_steps = tc["save_state_global_steps"],
      save_state_dir = tc["save_state_dir"],
      resume_dir=tc["resume_dir"],
      valid_strategy=tc["valid_strategy"],
      valid_steps=tc["valid_steps"],
      tb_writer=tb_writer,
      tb_record_global_steps=tc["tb_record_global_steps"])

# accelerate launch --config_file="../configs/accelerate_config.yaml" main.py
