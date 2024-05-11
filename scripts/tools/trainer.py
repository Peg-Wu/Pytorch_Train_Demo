import os
import torch
import pickle
from typing import Literal, Union
from .utils import Accumulator
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def _pickle_save(obj, save_path):
    with open(save_path, "wb") as f:
        pickle.dump(obj, f)


def _pickle_load(load_path):
    with open(load_path, "rb") as f:
        obj = pickle.load(f)
    return obj


def _calc_accuracy(outputs: torch.Tensor, 
                   targets: torch.Tensor,
                   Normalize: bool = False):
    
    num_true_preds = torch.eq(outputs.argmax(dim=-1), targets).sum().item()

    if Normalize:
        return num_true_preds / len(targets)
    else:
        return num_true_preds


_train_accumulator = Accumulator(n=3)  # [loss, num_true, pred_num_samples]
_valid_accumulator = Accumulator(n=2)  # [loss, num_true]


@torch.no_grad()
def _valid(model,
           valid_dl: DataLoader,
           loss_fn: torch.nn.CrossEntropyLoss,
           accelerator: Accelerator,
           tb_writer: SummaryWriter,
           current_global_step: int,
           valid_strategy: Literal["epoch", "step"],
           valid_steps: int):

    model.eval()
    _valid_accumulator.reset()
    for inputs, targets in valid_dl:
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        # gather all gpus' outputs and targets of one batch
        batch_outputs, batch_targets = accelerator.gather_for_metrics((outputs, targets))

        # calculate loss [reduction="sum"], num_true, and pred_num_samples of one batch
        batch_loss = accelerator.reduce(loss, reduction="sum")
        batch_num_true = _calc_accuracy(batch_outputs, batch_targets, Normalize=False)

        # accmulate them in `_valid_accumulator` [loss, num_true]
        _valid_accumulator.add(batch_loss, batch_num_true)

    # f"{xxx:>width}" (left padding) or f"{xxx:<width}" (right padding)
    accelerator.print(f"-" * 120 + "\n",
                      f"[Valid]\t",
                      f"Loss: {round(_valid_accumulator[0] / len(valid_dl.dataset), 8):<15}\t",
                      f"Accuracy: {round(_valid_accumulator[1] / len(valid_dl.dataset), 8):<10}" + "\n",
                      f"-" * 120)

    # write to tensorboard
    if accelerator.is_main_process:
        tb_writer.add_scalar(f"valid/loss_record_every_{valid_steps}_{valid_strategy}", 
                             _valid_accumulator[0] / len(valid_dl.dataset), current_global_step)
        tb_writer.add_scalar(f"valid/accuracy_record_every_{valid_steps}_{valid_strategy}", 
                             _valid_accumulator[1] / len(valid_dl.dataset), current_global_step)


def train(epochs: int,
          model,
          is_peft_model: bool,
          train_dl: DataLoader,
          valid_dl: DataLoader,
          loss_fn: torch.nn.CrossEntropyLoss,
          optimizer: torch.optim.Adam,
          accelerator: Accelerator,
          print_logs_global_steps: int,
          save_state_global_steps: int,
          save_state_dir: str,  # checkpoints/
          resume_dir: Union[str, Literal[-1]],  # checkpoints/global_step_xxx/
          valid_strategy: Literal["epoch", "step"],
          valid_steps: int,        
          tb_writer: SummaryWriter,
          tb_record_global_steps: int):
    
    global _train_accumulator

    # Default(-1): tb_record_global_steps = print_logs_global_steps
    tb_record_global_steps = [print_logs_global_steps if tb_record_global_steps == -1 else tb_record_global_steps][0]
    
    # modify loss reduction mode
    loss_fn.reduction = "sum"
    
    global_step = 0
    resume_epoch = 0

    # calculate #epochs and #steps needed to skip
    if isinstance(resume_dir, str):
        # load state
        accelerator.load_state(resume_dir)
        accelerator.print(f"resume from checkpoint -> {resume_dir}")
        global_steps_per_epoch = len(train_dl)
        has_trained_global_steps = int(resume_dir.split("global_step_")[-1])
        resume_epoch = has_trained_global_steps // global_steps_per_epoch  # resume epoch
        resume_step = has_trained_global_steps - resume_epoch * global_steps_per_epoch  # resume step
        global_step = has_trained_global_steps  # global_step = has_trained_global_steps

    for epoch in range(resume_epoch, epochs):
        model.train()
        _train_accumulator.reset()

        # skip_first_batches
        if isinstance(resume_dir, str) and epoch == resume_epoch and resume_step != 0:
            skipped_train_dl = accelerator.skip_first_batches(train_dl, num_batches=resume_step)
            # load accumulator
            _train_accumulator = _pickle_load(os.path.join(resume_dir, "train_accumulator.pkl"))
        else:
            skipped_train_dl = train_dl

        for inputs, targets in skipped_train_dl:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            accelerator.backward(loss)
            optimizer.step()
            global_step += 1

            # calculate training metrics (each gpu has trained one batch of data)
            with torch.no_grad():
                # gather all gpus' outputs and targets of one batch
                batch_outputs, batch_targets = accelerator.gather_for_metrics((outputs, targets))

                # calculate loss [reduction="sum"], num_true, and pred_num_samples of one batch
                batch_loss = accelerator.reduce(loss, reduction="sum")
                batch_num_true = _calc_accuracy(batch_outputs, batch_targets, Normalize=False)
                batch_pred_num_samples = len(batch_outputs)

                # accmulate them in `_train_accumulator` [loss, num_true, pred_num_samples]
                _train_accumulator.add(batch_loss, batch_num_true, batch_pred_num_samples)

                # print train_info (In every epoch, [step3] loss == [step1 and step2] avg_loss)
                if global_step % print_logs_global_steps == 0:
                    # f"{xxx:>width}" (left padding) or f"{xxx:<width}" (right padding)
                    accelerator.print(f"[Train]\t",
                                      f"Epoch: {epoch + 1:<5}\t",
                                      f"Global_Step: [{global_step:>10} / {len(train_dl) * epochs:<10}]\t\t"
                                      f"Loss: {round(_train_accumulator[0] / _train_accumulator[2], 8):<15}\t",
                                      f"Accuracy: {round(_train_accumulator[1] / _train_accumulator[2], 8):<10}")
                    
                # write to tensorboard
                if accelerator.is_main_process and global_step % tb_record_global_steps == 0:
                    tb_writer.add_scalar(f"train/loss_record_every_{tb_record_global_steps}_global_steps", 
                                         _train_accumulator[0] / _train_accumulator[2], global_step)
                    tb_writer.add_scalar(f"train/accuracy_record_every_{tb_record_global_steps}_global_steps", 
                                         _train_accumulator[1] / _train_accumulator[2], global_step)
                    
                    for name, param in model.named_parameters():
                        if param.requires_grad == True:
                            tb_writer.add_histogram(name + "/param", param, global_step)
                            tb_writer.add_histogram(name + "/grad", param.grad, global_step)
                
                # save state, accumulator, and adaptor
                if global_step % save_state_global_steps == 0:
                    # save state
                    accelerator.print(f"save checkpoint -> global_step_{global_step}")
                    accelerator.save_state(os.path.join(save_state_dir, f"global_step_{global_step}"))

                    # save accumulator
                    _pickle_save(_train_accumulator, os.path.join(save_state_dir, f"global_step_{global_step}", "train_accumulator.pkl"))

                    if is_peft_model:
                        # save model_adaptor (peft_model)
                        accelerator.unwrap_model(model).save_pretrained(
                            save_directory=os.path.join(save_state_dir, f"global_step_{global_step}", "adaptor"),
                            is_main_process=accelerator.is_main_process,
                            state_dict=accelerator.get_state_dict(model),
                            save_func=accelerator.save
                        )
                    
            # justify whether to valid
            if valid_strategy == "step" and global_step % valid_steps == 0:
                _valid(model, valid_dl, loss_fn, accelerator, tb_writer, global_step, valid_strategy, valid_steps)

        # justify whether to valid      
        if valid_strategy == "epoch" and (epoch + 1) % valid_steps == 0:
            _valid(model, valid_dl, loss_fn, accelerator, tb_writer, global_step, valid_strategy, valid_steps)

    # don't forget to close 
    tb_writer.close()
