#!/usr/bin/env python3
import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchsummary import summary
import wandb
from archive.unet import UNet
from model.unet_base import UNet_base
from model.smp_models import SegModel
from utils.data_utils import prepare_datasets
from utils.train_eval import train_fun, eval_fun
from utils.metrics import XEDiceLoss
from utils.metrics import setup_seed
import argparse
import importlib
import shutil
def train_model(
    model,
    device,
    epochs: int = 5,
    batch_size: int = 1,
    learning_rate: float = 1e-5,
    val_percent: float = 0.1,
    save_checkpoint: bool = True,
    img_scale: float = 0.5,
    amp: bool = False,
    weight_decay: float = 1e-8,
    momentum: float = 0.999,
    gradient_clipping: float = 1.0,
):
    # 1. Create dataset
    train_set, valid_set = prepare_datasets()

    # 3. Create data loaders
    loader_args = dict(
        batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True
    )
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(valid_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(
        project="Cloud Segmentation - Solafune Contest 2023",
        resume="allow",
        anonymous="must",
    )
    experiment.config.update(
        dict(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            val_percent=val_percent,
            save_checkpoint=save_checkpoint,
            img_scale=img_scale,
            amp=amp,
        )
    )

    logging.info(
        f"""Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {len(train_set)}
        Validation size: {len(valid_set)}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Mixed Precision: {amp}
    """
    )

    optimizer = optim.RMSprop(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        momentum=momentum,
        foreach=True,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", patience=5
    )  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = XEDiceLoss()
    global_step = 0
    best_iou = 0

    for epoch in range(1, epochs + 1):
        average_epoch_loss, global_step = train_fun(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            grad_scaler=grad_scaler,
            train_loader=train_loader,
            device=device,
            amp=amp,
            gradient_clipping=gradient_clipping,
            epoch=epoch,
            epochs=epochs,
            experiment=experiment,
            global_step=global_step,
        )
        print(f"Epoch {epoch}, Average Train Loss: {average_epoch_loss}")

        # Evaluation
        division_step = len(train_loader.dataset) // (5 * batch_size)
        if division_step > 0:
            if global_step % division_step == 0:
                histograms = {}
                for tag, value in model.named_parameters():
                    tag = tag.replace("/", ".")
                    if not (torch.isinf(value) | torch.isnan(value)).any():
                        histograms["Weights/" + tag] = wandb.Histogram(value.data.cpu())
                    if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                        histograms["Gradients/" + tag] = wandb.Histogram(
                            value.grad.data.cpu()
                        )

                iou, f1_score = eval_fun(model, val_loader, device, amp)
                scheduler.step(iou)

                logging.info("Valid iou: {}".format(iou))
                logging.info("Valid f1_score: {}".format(f1_score))
                try:
                    experiment.log(
                        {
                            "learning rate": optimizer.param_groups[0]["lr"],
                            "step": global_step,
                            "epoch": epoch,
                            **histograms,
                        }
                    )
                except:
                    pass
        if iou > best_iou:
            dir_checkpoint = Path("./checkpoint_inside/")
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            best_iou = iou
            torch.save(model.state_dict(), dir_checkpoint / f"best_model.pth")
            logging.info(f"Checkpoint {epoch} saved !")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}")
    
    # set random seed
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('-c', '--config', type=str, help='Configuration File')
    config_name=parser.parse_args().config
    config = importlib.import_module("." + config_name, package='config').config
    setup_seed(config['seed'])
    Kfold=config['Kfold_index']#0,1,2,3,4
    
    
    model = SegModel(encoder=config['encoder'], network=config['model_network'],
                     in_channels=config['in_channels'], n_class=config['n_class'])
    model.to(device=device)

    logging.info(f'Network:\n'
                f'\t{model.in_channels} input channels\n'
                f'\t{model.n_classes} output channels (classes)\n')
    
    
    # model save path
    save_ckpt_path = os.path.join('./checkpoints', config['save_path'], 'pth')
    save_log_path = os.path.join('./checkpoints', config['save_path'])
    if not os.path.exists(save_ckpt_path):
        os.makedirs(save_ckpt_path)
    if not os.path.exists(save_log_path):
        os.makedirs(save_log_path)
    config['save_log_path'] = save_log_path
    old_config_name_path='./config'+'/'+config_name+'.py'
    new_config_name_path = config['save_log_path'] + '/' + config_name + '.py'
    shutil.copyfile(src=old_config_name_path,dst=new_config_name_path)
    #copy the config.py to the log path
    config['save_ckpt_path'] = save_ckpt_path
    

    
    
    train_model(
        model=model,
        epochs=25,
        batch_size=4,
        learning_rate=1e-4,  # 0.0001
        device=device,
        img_scale=0.5,
        val_percent=10.0 / 100,
        amp=False,
    )
