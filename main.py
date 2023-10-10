#!/usr/bin/env python3
import argparse
import logging
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from model.smp_models import SegModel
from utils.data_utils import prepare_train_valid_dataset, prepare_full_dataset, LoadDataset, DataTransform
from utils.train_eval import train_fun, eval_fun
from utils.metrics import XEDiceLoss
from utils.metrics import get_seed
import importlib
from sklearn.model_selection import KFold
import os
import logging
import argparse
import torch
import importlib
# Import other necessary modules
from config.config import net_config, config, generate_save_path

def setup_logging(log_file="training.log"):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger()
    # Add a file handler to the logger
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(file_handler)
    return logger

def train_model(
    config,
    # model,
    net_config,
    device,
    epochs: int = 5,
    batch_size: int = 1,
    learning_rate: float = 1e-5,
    val_percent: float = 0.1,
    save_checkpoint: bool = True,
    amp: bool = False,
    weight_decay: float = 1e-8,
    gradient_clipping: float = 1.0,
):
    # 1. Create and load dataset
    full_data = prepare_full_dataset()
    
    loader_args = dict(
        num_workers=os.cpu_count(), pin_memory=True
    )
    # train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    # val_loader = DataLoader(valid_set, shuffle=False, drop_last=True, **loader_args)

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
            amp=amp,
        )
    )

    logging.info(
        f"""Starting training:
        in_channels {config["in_channels"]}
        n_class {config["n_class"]}
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Dataset size:   {len(full_data)}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Mixed Precision: {amp}
    """
    )


    # training
    global_step = 0
    num_folds = 4
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=config["seed"])
    for fold, (train_idx, val_idx) in enumerate(kf.split(full_data)):
        fold_config = net_config[fold]  # Get the configuration for this fold
        print("fold_config", fold_config)
        config['save_path'] = generate_save_path(fold)
        model = SegModel(
            encoder=fold_config["encoder"],
            network=fold_config["model_network"],
            in_channels=config["in_channels"],
            n_class=config["n_class"],
        )
        model.to(device=device)
        # model = nn.DataParallel(model)
        logging.info(f"Model: {fold_config['model_network']}")
        logging.info(f"Encoder: {fold_config['encoder']}")
        
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", patience=5)
        grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
        criterion = XEDiceLoss()
        best_f1 = 0

        print("Full Data Size: ", len(full_data))
        print("train_idx", len(train_idx))
        print("val_idx", len(val_idx))
        train_df = full_data.iloc[train_idx].reset_index(drop=True)
        val_df = full_data.iloc[val_idx].reset_index(drop=True)
        
        train_dataset = LoadDataset(df=train_df, phase='train', transform=DataTransform())
        valid_dataset = LoadDataset(df=val_df, phase='valid', transform=DataTransform())
        print(f"Fold: {fold}, Train Size: {len(train_df)}, Validation Size: {len(val_df)}")

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, **loader_args)
        val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,drop_last=True, **loader_args)
        
        
        for epoch in range(1, epochs + 1):
            logging.info(f"Epoch {epoch}/{epochs}")
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
            if global_step % len(train_loader) == 0: # evaluate every epoch
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
                print("iou", iou)
                
                
                scheduler.step(f1_score)

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
                if f1_score > best_f1:
                    save_ckpt_path = config["save_ckpt_path"]
                    best_f1 = f1_score
                    torch.save(
                        model.state_dict(), os.path.join(save_ckpt_path, f"best_{fold_config['model_network']}_{fold_config['encoder']}fold{fold}.pth")
                    )
                    logging.info(f"Best Checkpoint at {epoch} saved in fold {fold}!")
        
        print(f'Finished fold {fold+1}/{num_folds}')



# Add the setup_logging function
def setup_logging(log_file="training.log"):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger()
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(file_handler)
    return logger

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Train")
    parser.add_argument("-c", "--config", type=str, help="Configuration File")
    config_name = parser.parse_args().config
    
    # Import configuration and set seed for reproducibility
    config = importlib.import_module("." + config_name, package="config").config
    net_config = importlib.import_module("." + config_name, package="config").net_config
    # get_seed(config["seed"]) 
    

    
    # Setup model save path
    save_ckpt_path = os.path.join("./checkpoints", config["save_path"], "pth")
    if not os.path.exists(save_ckpt_path):
        os.makedirs(save_ckpt_path)
    config["save_ckpt_path"] = save_ckpt_path

    # Setup logging
    log_file_path = os.path.join(config["save_ckpt_path"], "training.log")
    logger = setup_logging(log_file=log_file_path)

    # Get device for training (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Log the starting of training and model details
    logger.info(f"Using device {device}")
    
    # Train the model
    train_model(
        config=config,
        # model=model,
        net_config= net_config,
        epochs=25,  # Adjust the number of epochs as needed
        batch_size=8,
        learning_rate=config["learning_rate"],
        device=device,
        val_percent=10.0 / 100,
        amp=False,
    )
