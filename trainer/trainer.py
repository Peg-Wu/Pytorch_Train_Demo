import os
import math
import peft
import torch
from .utils import *
from torch.utils.tensorboard import SummaryWriter

@torch.inference_mode()
def _valid(model, valid_dl, loss_fn, accelerator, valid_accumulator):
    
    model.eval()
    valid_accumulator.reset()

    for batch in valid_dl:
        inputs, targets = batch
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        # gather all gpus' outputs and targets of one batch
        all_proc_outputs, all_proc_targets = accelerator.gather_for_metrics((outputs, targets))

        # calculate loss and accuracy
        all_proc_loss = accelerator.reduce(loss, reduction="sum").item()
        all_proc_num_true = calc_accuracy(all_proc_outputs, all_proc_targets, Normalize=False)
        all_proc_num_pred_samples = len(all_proc_targets)

        # accmulate them in `valid_accumulator`
        valid_accumulator.add(all_proc_loss, all_proc_num_true, all_proc_num_pred_samples)

    return valid_accumulator

def train(epochs,
          model,
          train_dl,
          valid_dl,
          loss_fn,
          optimizer,
          accelerator,
          valid_strategy = "epoch",
          valid_interval = 1,
          print_train_logs_strategy = "epoch",
          print_train_logs_interval = 1,
          save_state_global_step = None,
          resume_dir = None,
          early_stop = None,
          tb_log_dir = None):
    
    """Train Function. By default, the model will be saved in the ./checkpoints/training_end_model after training ends.

    Parameters
    ----------
    epochs
        1. Train epochs.
    model
        1. Pytorch model.
    train_dl
        1. Train dataloader.
    valid_dl
        1. Valid dataloader.
    loss_fn
        1. Loss function, e.g. `nn.CrossEntropyLoss()`.
    optimizer
        1. Optimizer, e.g. `torch.optim.Adam()`.
    accelerator
        1. `accelerate.Accelerator()`.
    valid_strategy: str
        1. 'epoch' or 'global_step'.
    valid_interval: int
        1. Use together with `valid_stategy`. 
        2. if `valid_strategy`='epoch' and `valid_interval`=1, 
           this means that the model will be validated after training for one epoch.
    print_train_logs_strategy: str
        1. 'epoch' or 'global_step'.
    print_train_logs_interval: int
        1. Use together with `print_train_logs_strategy`. 
        2. if `print_train_logs_strategy`='epoch' and `print_train_logs_interval`=1, 
           this means that the model will print logs after each epoch of training.
    save_state_global_step
        1. int or None.
        2. Save train state every `save_state_global_step`
        3. Path to save state: `./checkpoints/global_step_xxx`
    resume_dir
        1. 'Path to resume_dir' or None.
        2. If resume_dir is specified, the model will resume training from this location.
        3. If resume_dir is None, the model will train from the begining.
    early_stop: int
        1. If the loss on the validation set does not decrease after early_stop epochs, training will be terminated early.
        2. If the early_stop parameter is not None, 
           the best model will be automatically saved in the ./checkpoints/early_stop_best_model/ directory.
        3. If early_stop is not None, then checkpoint resumption is not supported at this time.
    tb_log_dir
        1. Tensorboard log directory or None.

        
    examples:
        1. `train_large_model` :
        >>> train(epochs = 3,
                  model = model,
                  train_dl = train_dl,
                  valid_dl = valid_dl,
                  loss_fn = loss_fn,
                  optimizer = optimizer,
                  accelerator = accelerator,
                  valid_strategy = "global_step",
                  valid_interval = 10_000,
                  print_train_logs_strategy = "global_step",
                  print_train_logs_interval = 10_000,
                  save_state_global_step = 50_000,
                  resume_dir = None,
                  tb_log_dir = "./logs/")
        
        2. `train_small_model` :
        >>> train(epochs = 100,
                  model = model,
                  train_dl = train_dl,
                  valid_dl = valid_dl,
                  loss_fn = loss_fn,
                  optimizer = optimizer,
                  accelerator = accelerator,
                  early_stop = 8,
                  tb_log_dir = "./logs/")
    """

    # makedirs
    os.makedirs("./checkpoints/training_end_model", exist_ok=True)
    if early_stop is not None:
        os.makedirs("./checkpoints/early_stop_best_model", exist_ok=True)

    # create tensorboard
    if tb_log_dir is not None and accelerator.is_main_process:
        writer = SummaryWriter(log_dir=tb_log_dir)  # only main process has tensorboard writer

    # modify loss reduction mode
    loss_fn.reduction = "sum"

    # init early_stop
    if early_stop is not None:
        early_stop_counts, best_valid_loss = 0, math.inf
    
    # init global_step and resume_epoch
    global_step, resume_epoch = 0, 0

    # init accumulator
    train_accumulator = Accumulator(3)  # (loss, num_pred_true, num_pred_samples)
    valid_accumulator = Accumulator(3)  # (loss, num_pred_true, num_pred_samples)

    # calculate #epoch and #global_step needed to skip
    if resume_dir is not None:
        # load state
        accelerator.load_state(resume_dir)
        accelerator.print(f"resume from checkpoint -> {resume_dir}")
        global_steps_per_epoch = math.ceil(len(train_dl) / accelerator.gradient_accumulation_steps)
        has_trained_global_steps = int(resume_dir.split("global_step_")[-1])
        resume_epoch = has_trained_global_steps // global_steps_per_epoch  # resume epoch
        resume_step = has_trained_global_steps - global_steps_per_epoch * resume_epoch  # resume step
        global_step = has_trained_global_steps  # reinitialize global_step

    for epoch in range(resume_epoch, epochs):
        model.train()
        train_accumulator.reset()
    
        # skip_first_batches
        if resume_dir is not None and epoch == resume_epoch and resume_step != 0:
            skipped_train_dl = accelerator.skip_first_batches(train_dl, num_batches=resume_step * accelerator.gradient_accumulation_steps)
            # load train accumulator
            train_accumulator = load_from_pickle(os.path.join(resume_dir, "train_accumulator.pkl"))
        else:
            skipped_train_dl = train_dl
        
        for batch in skipped_train_dl:
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                inputs, targets = batch
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                accelerator.backward(loss)
                optimizer.step()

                if accelerator.sync_gradients:
                    global_step += 1

            # calculate training metrics (each gpu has trained one batch of data)
            with torch.no_grad():
            # gather all gpus' outputs and targets of one batch
                all_proc_outputs, all_proc_targets = accelerator.gather_for_metrics((outputs, targets))

                # calculate loss and accuracy
                all_proc_loss = accelerator.reduce(loss, reduction="sum").item()
                all_proc_num_true = calc_accuracy(all_proc_outputs, all_proc_targets, Normalize=False)
                all_proc_num_pred_samples = len(all_proc_targets)

                # accmulate them in `train_accumulator`
                train_accumulator.add(all_proc_loss, all_proc_num_true, all_proc_num_pred_samples)

                # print train logs
                if print_train_logs_strategy == "global_step" and global_step % print_train_logs_interval == 0:
                    accelerator.print(f"[Train]\t",
                                        f"epoch: {epoch + 1}/{epochs}\t",
                                        f"global_step: {global_step}/{len(train_dl) // accelerator.gradient_accumulation_steps * epochs}\t",
                                        f"loss: {train_accumulator[0] / train_accumulator[2]}\t",
                                        f"accuracy: {train_accumulator[1] / train_accumulator[2]}")
                    
                    # write to tensorboard, x-axis is global_step
                    if tb_log_dir is not None and accelerator.is_main_process:
                        writer.add_scalar("train/loss", train_accumulator[0] / train_accumulator[2], global_step)
                        writer.add_scalar("train/accuracy", train_accumulator[1] / train_accumulator[2], global_step)

                        for name, param in model.named_parameters():
                            if param.requires_grad == True:
                                writer.add_histogram(name + "/grad", param.grad, global_step)

                # justify whether to valid
                if valid_strategy == "global_step" and global_step % valid_interval == 0:
                    valid_accumulator = _valid(model, valid_dl, loss_fn, accelerator, valid_accumulator)

                    # print validation result
                    accelerator.print(f"[Valid]\t",
                                    f"epoch: {epoch + 1}/{epochs}\t",
                                    f"global_step: {global_step}/{len(train_dl) // accelerator.gradient_accumulation_steps * epochs}\t",
                                    f"loss: {valid_accumulator[0] / valid_accumulator[2]}\t",
                                    f"accuracy: {valid_accumulator[1] / valid_accumulator[2]}")
                
                    # write to tensorboard, x-axis is global_step
                    if tb_log_dir is not None and accelerator.is_main_process:
                        writer.add_scalar("valid/loss", valid_accumulator[0] / valid_accumulator[2], global_step)
                        writer.add_scalar("valid/accuracy", valid_accumulator[1] / valid_accumulator[2], global_step)
                
                # save state, accumulator
                if save_state_global_step is not None and global_step % save_state_global_step == 0:
                    # save state
                    accelerator.print(f"save checkpoint -> global_step_{global_step}")
                    accelerator.save_state(os.path.join("./checkpoints", f"global_step_{global_step}"))

                    # save train_accumulator
                    save_to_pickle(train_accumulator, os.path.join("./checkpoints", f"global_step_{global_step}", "train_accumulator.pkl"))

        # print train logs
        if print_train_logs_strategy == "epoch" and (epoch + 1) % print_train_logs_interval == 0:
            accelerator.print(f"[Train]\t",
                              f"epoch: {epoch + 1}/{epochs}\t",
                              f"global_step: {global_step}/{len(train_dl) // accelerator.gradient_accumulation_steps * epochs}\t",
                              f"loss: {train_accumulator[0] / train_accumulator[2]}\t",
                              f"accuracy: {train_accumulator[1] / train_accumulator[2]}")
            
            # write to tensorboard, x-axis is epoch
            if tb_log_dir is not None and accelerator.is_main_process:
                writer.add_scalar("train/loss", train_accumulator[0] / train_accumulator[2], epoch + 1)
                writer.add_scalar("train/accuracy", train_accumulator[1] / train_accumulator[2], epoch + 1)

                for name, param in model.named_parameters():
                    if param.requires_grad == True:
                        writer.add_histogram(name + "/grad", param.grad, global_step)

        # justify whether to valid      
        if valid_strategy == "epoch" and (epoch + 1) % valid_interval == 0:
            valid_accumulator = _valid(model, valid_dl, loss_fn, accelerator, valid_accumulator)

            # print validation result
            accelerator.print(f"[Valid]\t",
                              f"epoch: {epoch + 1}/{epochs}\t",
                              f"global_step: {global_step}/{len(train_dl) // accelerator.gradient_accumulation_steps * epochs}\t",
                              f"loss: {valid_accumulator[0] / valid_accumulator[2]}\t",
                              f"accuracy: {valid_accumulator[1] / valid_accumulator[2]}")

            # write to tensorboard, x-axis is epoch
            if tb_log_dir is not None and accelerator.is_main_process:
                writer.add_scalar("valid/loss", valid_accumulator[0] / valid_accumulator[2], epoch + 1)
                writer.add_scalar("valid/accuracy", valid_accumulator[1] / valid_accumulator[2], epoch + 1)
            
            # early_stop
            if early_stop is not None:
                current_epoch_loss = valid_accumulator[0] / valid_accumulator[2]
                if current_epoch_loss < best_valid_loss:
                    best_valid_loss = current_epoch_loss

                    # save the early_stop_best_model, if is_peft_model, only save adaptor
                    if isinstance(model, peft.peft_model.PeftModel):
                        accelerator.unwrap_model(model).save_pretrained(
                            save_directory="./checkpoints/early_stop_best_model",
                            is_main_process=accelerator.is_main_process,
                            state_dict=accelerator.get_state_dict(model),
                            save_func=accelerator.save)
                    else:
                        accelerator.wait_for_everyone()
                        accelerator.save(accelerator.get_state_dict(model), "./checkpoints/early_stop_best_model/pytorch_model.bin")
                    
                    early_stop_counts = 0
                    accelerator.print(f"epoch {epoch + 1} reach a lower validation loss of {best_valid_loss}, saving model to ./checkpoints/early_stop_best_model!")
                else:
                    early_stop_counts += 1
                
                if early_stop_counts >= early_stop:
                    accelerator.print(f"model is not improving in {early_stop} epochs, so we halt the training session!")
                    break

    # don't forget to close tensorboard writer
    if accelerator.is_main_process:
        writer.close()
    
    # save the final model after training, if is_peft_model, only save adaptor
    if isinstance(model, peft.peft_model.PeftModel):
        accelerator.unwrap_model(model).save_pretrained(
            save_directory="./checkpoints/training_end_model",
            is_main_process=accelerator.is_main_process,
            state_dict=accelerator.get_state_dict(model),
            save_func=accelerator.save)
    else:
        accelerator.wait_for_everyone()
        accelerator.save(accelerator.get_state_dict(model), "./checkpoints/training_end_model/pytorch_model.bin")

@torch.inference_mode()
def test(model, test_dl, loss_fn, accelerator):

    """Test Function. By default, test loss and accuracy will be printed.

    Parameters
    ----------
    model
        1. pytorch model
    test_dl
        1. test dataloader
    loss_fn
        1. loss function
    accelerator
        1. accelerator = accelerate.Accelerator()
    """
    
    # modify loss reduction mode
    loss_fn.reduction = "sum"

    model.eval()
    test_accumulator = Accumulator(3)  # (loss, num_pred_true, num_pred_samples)

    for batch in test_dl:
        inputs, targets = batch
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        # gather all gpus' outputs and targets of one batch
        all_proc_outputs, all_proc_targets = accelerator.gather_for_metrics((outputs, targets))

        # calculate loss and accuracy
        all_proc_loss = accelerator.reduce(loss, reduction="sum").item()
        all_proc_num_true = calc_accuracy(all_proc_outputs, all_proc_targets, Normalize=False)
        all_proc_num_pred_samples = len(all_proc_targets)

        # accmulate them in `valid_accumulator`
        test_accumulator.add(all_proc_loss, all_proc_num_true, all_proc_num_pred_samples)

    accelerator.print(f"[Test]\t",
                      f"loss: {test_accumulator[0] / test_accumulator[2]}\t",
                      f"accuracy: {test_accumulator[1] / test_accumulator[2]}")