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
def poly_lr(epoch, max_epochs, initial_lr, exponent=0.9):
    return initial_lr * (1 - epoch / max_epochs) ** exponent
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
    
    # from torch.nn import CrossEntropyLoss
    # Move model to the specified device
    model = model.to(device)

    # Define the criterion (loss function) and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Learning rate scheduler (without verbose)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: poly_lr(epoch, num_epochs, 1e-3) / 1e-3)
    # criterion = CrossEntropyLoss(weight=torch.Tensor([0.001]))
    min_loss = np.inf  # Initialize minimum loss to positive infinity

    for epoch in range(num_epochs):
        train_loss = 0.0

        # Update the training dataset every 10 epochs
        for patient_id in train_list:
            train_dataset = AneurysmDataset(str(patient_id))
            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

            # Training Phase
            model.train()
            for batch in train_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                # Forward pass
                outputs = model(inputs)
                # Compute loss
                loss = loss_fn(outputs, targets)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                # Calculate Precision for label 1
                
                # print(f'final_matrix shape: {final_matrix.shape}, targets shape: {targets.shape}')
                # if epoch % 5 == 4 and torch.sum(targets) > 0:
                #     final_matrix = F.sigmoid(outputs).float()
                #     final_matrix = (final_matrix>0.5).float()
                #     true_positives = torch.sum((final_matrix == 1) & (targets == 1))
                #     false_positives = torch.sum((final_matrix == 1) & (targets == 0))
                #     false_negatives = torch.sum((final_matrix == 0) & (targets == 1))
                #     true_negatives = torch.sum((final_matrix == 0) & (targets == 0))
                #     recall = true_positives / (true_positives + false_negatives + 1e-8)
                #     precision = true_positives / (true_positives + false_positives + 1e-8)
                #     print(f'Number of label: {torch.sum(targets)}')
                #     print(f'True positive: {true_positives}')
                #     print(f'false positive: {false_positives}')
                #     print(f'false negative: {false_negatives}')
                #     print(f'true negative: {true_negatives}')
                #     print(f'Precision: {precision}')
                #     print(f'Recall: {recall}')
            del train_dataset  # Free memory
            del train_loader
        train_loss /= len(train_list)
        scheduler.step() 
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}")
        model_save_path = os.path.join('/kaggle/working/', 'best.pt')
        torch.save(model.module.state_dict(), model_save_path)
        # Validation Phase every 10 epochs
        if epoch % 20 == 19:
            val_loss = validate_numpy_cases(
                model=model,
                valid_list=valid_list,
                window_size=window_size,
                overlap=overlap,
                device=device,
            )
            print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss}")

            # Save the best model
            if val_loss < min_loss:
                min_loss = val_loss
                # model_save_path = os.path.join('/kaggle/working/', 'best.pt')
                # torch.save(model.module.state_dict(), model_save_path)
                # print(f"Model saved to {model_save_path}")

            # Step the scheduler
            # scheduler.step(val_loss)

            # Log the learning rate
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Learning Rate after Epoch {epoch + 1}: {current_lr:.6f}")

    print("Training complete.")

def validate_numpy_cases(model, valid_list, window_size=64, overlap=32, device = torch.device('cuda')):
    """
    Validate a 3D U-Net model on a list of 3D numpy arrays using a sliding window approach.

    Args:
        model (torch.nn.Module): The 3D U-Net model.
        valid_data (list of np.ndarray): List of 3D numpy arrays for input data.
        target_list (list of np.ndarray): List of 3D numpy arrays for ground truth.
        window_size (int): Sliding window size.
        overlap (int): Overlap size.
        device (str): Device for validation ('cpu' or 'cuda').

    Returns:
        float: Mean Dice loss over all cases.
    """
    
    model.eval()
    step = window_size-overlap
    total_loss = 0.0
    case_count = len(valid_list)

    with torch.no_grad():
        for case_idx, patient_id in enumerate(valid_list):
            print(patient_id)
            subfolder = os.path.join("/kaggle/input/pre-aneurysm-dataset/pre_aneurysm_dataset", patient_id)
            data = np.load(os.path.join(subfolder,'image.npy'))
            target = np.load(os.path.join(subfolder,'label.npy'))
            
            # Convert numpy arrays to tensors and add batch and channel dimensions
            data_tensor = torch.tensor(data).unsqueeze(0).unsqueeze(0).float().to(device)  # Shape: (1, 1, D, H, W)
            target_tensor = torch.tensor(target).unsqueeze(0).unsqueeze(0).float().to(device)  # Shape: (1, 1, D, H, W)
            h,w,d = data_tensor.shape[2], data_tensor.shape[3], data_tensor.shape[4]
            counting_matrix = torch.zeros(1, 1, h, w, d, device=device)
            prediction_matrix = torch.zeros(1, 1, h, w, d, device=device)
            sum_prediction_matrix = torch.zeros(1, 1, h, w, d, device=device)
            case_loss = 0.0
            count = 0
            #3D
            for y in range(0, h - window_size + 1, step):
                for x in range(0, w - window_size + 1, step):
                    for z in range(0, d - window_size + 1, step):
                        # Forward pass
                        predictions = F.sigmoid(model(data_tensor[:, :, x:x+window_size, y:y+window_size, z:z+window_size]))           
                        sum_prediction_matrix[:, :, x:x+window_size, y:y+window_size, z:z+window_size] += predictions
                        counting_matrix[:, :, x:x+window_size, y:y+window_size, z:z+window_size] += 1                   
                        loss = loss_fn(predictions, target_tensor[:, :, x:x+window_size, y:y+window_size, z:z+window_size])
                        case_loss += loss
                        count += 1
            #2D
            for y in range(0, h - window_size + 1, step):
                for x in range(0, w - window_size + 1, step):
                        # Forward pass
                        predictions = F.sigmoid(model(data_tensor[:, :, x:x+window_size, y:y+window_size, -64:])) 
                        sum_prediction_matrix[:, :, x:x+window_size, y:y+window_size, -64:] += predictions
                        counting_matrix[:, :, x:x+window_size, y:y+window_size, -64:] += 1
                        loss = loss_fn(predictions, target_tensor[:, :, x:x+window_size, y:y+window_size, -64:])
                        case_loss += loss
                        count += 1
            for y in range(0, h - window_size + 1, step):
                for z in range(0, d - window_size + 1, step):
                        # Forward pass
                        predictions = F.sigmoid(model(data_tensor[:, :, -64:, y:y+window_size:, z:z+window_size]))
                        # Get the predicted class (argmax across channels)
                        sum_prediction_matrix[:, :, -64:, y:y+window_size:, z:z+window_size] += predictions
                        counting_matrix[:, :, -64:, y:y+window_size, z:z+window_size] += 1
                        loss = loss_fn(predictions, target_tensor[:, :, -64:, y:y+window_size:, z:z+window_size])
                        case_loss += loss
                        # Compute Dice loss
                        count += 1
            for x in range(0, w - window_size + 1, step):
                for z in range(0, d - window_size + 1, step):
                        # Forward pass
                        predictions = F.sigmoid(model(data_tensor[:, :, x:x+window_size:, -64:, z:z+window_size]))
                        # Get the predicted class (argmax across channels)
                    # Compute Dice loss
                        sum_prediction_matrix[:, :, x:x+window_size, -64:, z:z+window_size] += predictions
                        counting_matrix[:, :, x:x+window_size, -64:, z:z+window_size] += 1    
                        loss = loss_fn(predictions, target_tensor[:, :, x:x+window_size, -64:, z:z+window_size])
                        case_loss += loss       
                        count += 1
            #1D
            for y in range(0, h - window_size + 1, step):
                        # Forward pass
                        predictions = F.sigmoid(model(data_tensor[:, :, -64:, y:y+window_size, -64:]))
                        sum_prediction_matrix[:, :, -64:, y:y+window_size, -64:] += predictions
                        counting_matrix[:, :,  -64:, y:y+window_size, -64:] += 1
                        # Compute Dice loss
                        loss = loss_fn(predictions, target_tensor[:, :, -64:, y:y+window_size, -64:])
                        case_loss += loss
                        count += 1
            for x in range(0, w - window_size + 1, step):
                        # Forward pass
                        predictions = F.sigmoid(model(data_tensor[:, :, x:x+window_size, -64:, -64:]))
                        sum_prediction_matrix[:, :, x:x+window_size, -64:, -64:] += predictions
                        counting_matrix[:, :, x:x+window_size, -64:, -64:] += 1
                        # Compute Dice loss
                        loss = loss_fn(predictions, target_tensor[:, :, x:x+window_size, -64:, -64:])
                        case_loss += loss
                        count += 1
            for z in range(0, d - window_size + 1, step):
                        # Forward pass
                        predictions = F.sigmoid(model(data_tensor[:, :, -64:, -64:, z:z+window_size]))
                        sum_prediction_matrix[:, :, -64:, -64:, z:z+window_size] += predictions
                        counting_matrix[:, :, -64:, -64:, z:z+window_size] += 1                    
                        # Compute loss   
                        loss = loss_fn(predictions, target_tensor[:, :, -64:, -64:, z:z+window_size])
                        case_loss += loss  
                        count += 1
            # Last block
            predictions = F.sigmoid(model(data_tensor[:, :, -64:, -64:, -64:]))
            sum_prediction_matrix[:, :, -64:, -64:, -64:] += predictions
            counting_matrix[:, :, -64:, -64:, -64:] += 1
            # Compute  loss
            loss = loss_fn(predictions, target_tensor[:, :, -64:, -64:, -64:])
            case_loss += loss            
            count += 1
            # Normalize loss for the case
            if count > 0:
                case_loss /= count

            total_loss += case_loss
            final_matrix = sum_prediction_matrix / counting_matrix
            final_matrix = (final_matrix > 0.5).float()
            # Calculate metrics
            true_positives = torch.sum((final_matrix == 1) & (target_tensor == 1))
            print(f'True positives{true_positives}')
            false_positives = torch.sum((final_matrix == 1) & (target_tensor == 0))
            print(f'True positives{false_positives}')
            false_negatives = torch.sum((final_matrix == 0) & (target_tensor == 1))
            true_negatives = torch.sum((final_matrix == 0) & (target_tensor == 0))
            # Calculate Precision for label 1
            precision = true_positives / (true_positives + false_positives + 1e-8)
            print(f'Precision: {precision}')
            # Accuracy            
            accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives + 1e-8)
            print(f'Accuracy: {accuracy}')
            # Dice coefficient
            dice_coefficent = 2 * true_positives / (2 * true_positives + false_positives + false_negatives + 1e-8)
            print(f'Dice coefficent: {dice_coefficent}')
            #Recall
            recall = true_positives / (true_positives + false_negatives + 1e-8)
            print(f'Recall: {recall}')
    # Compute the mean loss across all cases
    mean_loss = total_loss / case_count if case_count > 0 else float('inf')
    return mean_loss