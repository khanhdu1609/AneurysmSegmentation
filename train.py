from torch.optim.lr_scheduler import LambdaLR
from torch.nn.modules.loss import _Loss
from collections.abc import Callable, Sequence
import torch.nn as nn
from torch.utils.data import DataLoader
import random
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import os
import torch
import random
import numpy as np
from torch.optim import lr_scheduler
from skimage.util import montage
import matplotlib.pyplot as plt
import warnings
from enum import Enum
from dataset import AneurysmDataset
from unet_3d import UNet3D
from loss import loss_fn
import wandb
def train_log(loss, epoch, optimizer):
    # Get the current learning rate from the optimizer
    lr = optimizer.param_groups[0]['lr']

    # Log loss, epoch, and learning rate
    wandb.log({"epoch": epoch+1, "Train loss": loss, "learning_rate": lr})

    print(f"Epoch {epoch+1} - Loss: {loss:.3f} - Learning Rate: {lr:.6f}")

from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau, SequentialLR
def train_3d_unet(model, train_list, valid_list, num_epochs, window_size=64, overlap=32, device="cuda"):
    """
    Train a 3D U-Net model with Dice loss and validate using a sliding window approach.

    Args:
        model (torch.nn.Module): The 3D U-Net model.
        train_list (list): List of training data samples.
        valid_data (torch.Tensor): Validation data of shape (N, C, D, H, W).
        num_epochs (int): Number of training epochs.
        window_size (int): Sliding window size for validation.
        overlap (int): Overlap size for sliding window.
        device (str): Device for training ('cpu' or 'cuda').

    Returns:
        None
    """
    import torch
    import numpy as np
    import os
    from torch.utils.data import DataLoader
    import torch.optim as optim
    # from torch.nn import CrossEntropyLoss
    # Move model to the specified device
    model = model.to(device)

    # Define the criterion (loss function) and optimizer
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    #Linear warmup
    def warmup_lambda(epoch):
        if epoch < 25:
            return (epoch + 1) / 25 
        return 1.0
    warmup_scheduler = LambdaLR(optimizer, warmup_lambda)
    # Learning rate scheduler (without verbose)
    reduce_lr_scheduler  = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=40, factor=0.5, min_lr=1e-7)
    min_valid_loss = np.inf  # Initialize minimum loss to positive infinity
    wandb.watch(model, loss_fn, log='all', log_freq=1)
    for epoch in range(num_epochs):
        train_loss = 0.0

        # Update the training dataset every 10 epochs
        for patient_id in train_list:
            train_dataset = AneurysmDataset(str(patient_id))
            train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

            # Training Phase
            model.train()
            for batch in train_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                # Forward pass
                outputs, aux_out2, aux_out3, aux_out4 = model(inputs)
                main_loss = loss_fn(outputs, targets.to(device))
                deep_loss2 = loss_fn(aux_out2, F.interpolate(targets.float(), size = aux_out2.shape[2:]).to(device))
                deep_loss3 = loss_fn(aux_out3, F.interpolate(targets.float(), size = aux_out3.shape[2:]).to(device))
                deep_loss4 = loss_fn(aux_out4, F.interpolate(targets.float(), size = aux_out4.shape[2:]).to(device))
                loss = main_loss*8/15 + deep_loss2*4/15 + deep_loss3*2/15 + deep_loss4*1/15
                # if epoch < 5:
                #         final_matrix = F.sigmoid(outputs).float()
                #         final_matrix = (final_matrix>0.5).float()
                #         true_positives = torch.sum((final_matrix == 1) & (targets == 1))
                #         false_positives = torch.sum((final_matrix == 1) & (targets == 0))
                #         false_negatives = torch.sum((final_matrix == 0) & (targets == 1))
                #         true_negatives = torch.sum((final_matrix == 0) & (targets == 0))
            
                    
                #         print(f'{patient_id} TP {true_positives} FP {false_positives} FN {false_negatives}')

                # Compute loss
                # loss = loss_fn(outputs, targets)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            del train_dataset  # Free memory
            del train_loader
        train_loss /= len(train_list)
        train_log(train_loss, epoch, optimizer)
        

        valid_loss = 0
        valid_precision = 0
        valid_recall = 0

        for patient_id in valid_list:
                loss_case = 0
                valid_dataset = AneurysmDataset(str(patient_id), mode='valid')
                valid_loader = DataLoader(valid_dataset, batch_size=2, shuffle=True)
    
                #set to eval mode
                model.eval()
                with torch.no_grad():
                    for batch in valid_loader:
                        inputs, targets = batch
                        inputs, targets = inputs.to(device), targets.to(device)
                        # Forward pass
                        outputs, aux_out2, aux_out3, aux_out4 = model(inputs)
                        final_matrix = F.sigmoid(outputs).float()
                        final_matrix = (final_matrix>0.5).float()
                        true_positives = torch.sum((final_matrix == 1) & (targets == 1))
                        false_positives = torch.sum((final_matrix == 1) & (targets == 0))
                        false_negatives = torch.sum((final_matrix == 0) & (targets == 1))
                        true_negatives = torch.sum((final_matrix == 0) & (targets == 0))
                        recall = true_positives / (true_positives + false_negatives + 1e-8)
                        precision = true_positives / (true_positives + false_positives + 1e-8)
                        valid_precision += precision.item()
                        valid_recall += recall.item()

                        loss = loss_fn(outputs, targets)
                        loss_case += loss.item()
                loss_case /= len(valid_loader)
                valid_loss += loss_case
        valid_loss /= len(valid_list)
        if epoch < 25:
            warmup_scheduler.step()  # Warm-up trong 25 epoch đầu
        else:
            reduce_lr_scheduler.step(valid_loss) 
        wandb.log({"Valid loss": valid_loss, "Precision": valid_precision / len(valid_list), "Recall": valid_recall / len(valid_list)})
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Valid loss: {valid_loss:.4f}")
        if valid_loss < min_valid_loss:
            min_valid_loss = valid_loss
            model_save_path = os.path.join('/kaggle/working/', 'best_256_adam_5325.pt')
            torch.save({'model_state_dict': model.module.state_dict(), \
                        'optimizer_state_dict': optimizer.state_dict(), \
                        }, model_save_path)
            print(f'Model saved to {model_save_path}!')


    print("Training complete.")