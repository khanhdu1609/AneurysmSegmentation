import torch
import os
import numpy as np
from loss import loss_fn
import torch.nn.functional as F
window_size=np.asarray([256,256,64])
print(window_size/2)
def validate_numpy_cases(model, valid_list, window_size=np.asarray([256,256,64]), device = torch.device('cuda')):
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
    step = window_size/2
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
            print(f'max:{torch.max(final_matrix)}')
            final_matrix = (final_matrix > 0.5).float()
            # Calculate metrics
            true_positives = torch.sum((final_matrix == 1) & (target_tensor == 1))
            print(f'True positives{true_positives}')
            false_positives = torch.sum((final_matrix == 1) & (target_tensor == 0))
            print(f'False positives{false_positives}')
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