import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import random

def get_crop_coordinates(bbox, image_shape):
    # Calculate the block size along each axis, adjusted to the nearest multiple of 64
    len_block_x = ((bbox['x'][1] - bbox['x'][0]) // 64 + 1) * 64
    len_block_y = ((bbox['y'][1] - bbox['y'][0]) // 64 + 1) * 64
    len_block_z = ((bbox['z'][1] - bbox['z'][0]) // 64 + 1) * 64
        
    # Calculate coordinates along the x-axis
    distance_x = random.randint(0, len_block_x - (bbox['x'][1] - bbox['x'][0]))
    if bbox['x'][0] - distance_x < 0:
        start_x, end_x = 0, len_block_x
    elif bbox['x'][1] + 64 - distance_x > image_shape[0]:
        start_x, end_x = image_shape[0] - len_block_x, image_shape[0]
    else:
        start_x, end_x = bbox['x'][0] - distance_x, bbox['x'][0] - distance_x + len_block_x

    # Calculate coordinates along the y-axis
    distance_y = random.randint(0, len_block_y - (bbox['y'][1] - bbox['y'][0]))
    if bbox['y'][0] - distance_y < 0:
        start_y, end_y = 0, len_block_y
    elif bbox['y'][1] + 64 - distance_y > image_shape[1]:
        start_y, end_y = image_shape[1] - len_block_y, image_shape[1]
    else:
        start_y, end_y = bbox['y'][0] - distance_y, bbox['y'][0] - distance_y + len_block_y
    # Calculate coordinates along the z-axis
    distance_z = random.randint(0, len_block_z - (bbox['z'][1] - bbox['z'][0]))
    if bbox['z'][0] - distance_z < 0:
        start_z, end_z = 0, len_block_z
    elif bbox['z'][1] + 64 - distance_z > image_shape[2]:
        start_z, end_z = image_shape[2] - len_block_z, image_shape[2]
    else:
        start_z, end_z = bbox['z'][0] - distance_z, bbox['z'][0] - distance_z + len_block_z
        
    return (start_x, end_x), (start_y, end_y), (start_z, end_z)
    
import numpy as np

def split_3d(data_coor, block_size=(64, 64, 64)):
    """
    Splits a 3D NumPy array into 64x64x64 blocks.

    Args:
        array: The input 3D NumPy array.
        block_size: The size of each block along each dimension (default: 64).

    Returns:
        A list of 3D NumPy arrays, each representing a 64x64x64 block.
    """

    assert len(data_coor) == 3, "Input array must be 3D"
    shape0, shape1, shape2 = data_coor[0][1] - data_coor[0][0], data_coor[1][1] - data_coor[1][0], data_coor[2][1] - data_coor[2][0]
    assert shape0 % block_size[0] == 0, "Dimensions must be multiples of block_size"
    assert shape1 % block_size[1] == 0, "Dimensions must be multiples of block_size"
    assert shape2 % block_size[2] == 0, "Dimensions must be multiples of block_size"
    splitted_coors = []
    for i in range(0, shape0, block_size[0]):
        for j in range(0, shape1, block_size[1]):        
            for k in range(0, shape2, block_size[2]):
                splitted_coors.append(((data_coor[0][0]+i,data_coor[0][0]+i+block_size[0]), (data_coor[1][0]+j,data_coor[1][0]+j+block_size[1]), (data_coor[2][0]+k,data_coor[2][0]+k+block_size[2])))
                

    return splitted_coors
def get_random_crop_coordinates(image_shape, crop_size=(64, 64, 64)):
    """
    Calculates random starting coordinates for cropping a block of a given size
    from a 3D NumPy array.

    Args:
      image_shape: The shape of the input 3D array (tuple of three integers).
      crop_size: The size of the crop along each dimension (default is 64).

    Returns:
      A tuple of start and end indices for each dimension.
    """
    d0, d1, d2 = image_shape[0], image_shape[1], image_shape[2]
    if d0 < crop_size[0] or d1 < crop_size[1] or d2 < crop_size[2]:
        raise ValueError("Array dimensions must be at least equal to the crop size.")

    start_d0 = np.random.randint(0, d0 - crop_size[0] + 1)
    start_d1 = np.random.randint(0, d1 - crop_size[1] + 1)
    start_d2 = np.random.randint(0, d2 - crop_size[2] + 1)

    return (
        (start_d0, start_d0 + crop_size[0]),
        (start_d1, start_d1 + crop_size[1]),
        (start_d2, start_d2 + crop_size[2]),
    )
import re
def find_bounding_boxes(file_path):
  # Read the content from the file
  with open(file_path, "r") as file:
      text = file.read()

  # Regular expression to extract bounding box information
  pattern = r"Bounding Box (\d+) Coordinates:\s*x:\s*(\d+)-(\d+)\s*y:\s*(\d+)-(\d+)\s*z:\s*(\d+)-(\d+)"
  matches = re.findall(pattern, text)

  # Extract and structure the data
  bounding_boxes = []
  for match in matches:
      box_id = int(match[0])
      bounding_boxes.append({"Box ID": box_id, "x": (int(match[1]), int(match[2])), "y": (int(match[3]), int(match[4])), "z": (int(match[5]), int(match[6]))})
  return bounding_boxes

def get_split_coordinates(arr_shape):
    """
    Calculates the coordinates for splitting a 3D array into 4 equal parts
    along the first two dimensions, while keeping the full depth intact.

    Args:
      arr_shape: The shape of the input 3D array (tuple of three integers).

    Returns:
      A list of tuples, each containing start and end indices for the first
      two dimensions of each part.
    """
    d0, d1, d2 = arr_shape
    coordinates = d0, d1, d2
    return coordinates
# image_data = image_data.transpose(1, 2, 0)
# Get the image data as a NumPy array
# image_data = nii.get_fdata()
# print(nii.header.get_zooms())
# print(image_data.shape)
# # Choose a slice from the 3D volume to visualize
# slice_index = image_data.shape[2] // 2  # Select the middle slice
# for i in range(512):
#     for j in range(512):
#         print(image_data[i, j, 70])
# print(image_data[:, :, 70])
# Plot the selected slice
# Get the image data as a NumPy array
import os
import numpy as np
import random
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# Load the NIfTI file
class AneurysmDataset(Dataset):
    def __init__(self, patient_id, root_dir="C:/Users/XgearHN_Lap/Downloads/train_aneurysm_data"):
        self.root_dir = root_dir
        # image_data = np.load(file_path)
        # root_dir = "path_to_root_directory"  # Replace with actual root directory
        self.patient_id = patient_id  # Replace with actual patient ID
        patient_path = os.path.join(self.root_dir, self.patient_id)
        self.train_coordinates = []
        # self.transform = transforms.ToTensor()
        # Check if it's a directory
        if os.path.isdir(patient_path):
            image_file = os.path.join(patient_path, "image.npy")
            self.normalized_image_data = np.load(image_file)
            label_file = os.path.join(patient_path, "label.npy")
            self.label_data = np.load(label_file)
            data_shape = self.normalized_image_data.shape

            coor = get_split_coordinates(data_shape)
            print(coor[0])
            # Get random crop coordinates
            (start_d0, end_d0), (start_d1, end_d1), (start_d2, end_d2) = get_random_crop_coordinates(coor)
            self.train_coordinates.append(((start_d0, end_d0), (start_d1, end_d1), (start_d2, end_d2)))

            if 1 in self.label_data:
                bounding_boxes_file = os.path.join(patient_path, 'bounding_boxes.txt')
                bounding_boxes = find_bounding_boxes(bounding_boxes_file)
                labeled_coordinates = []

                for bbox in bounding_boxes:
                    if bbox is not None:
                        bbox_coordinate = get_crop_coordinates(bbox, data_shape)
                        splitted_coordinates = split_3d(bbox_coordinate)
                        labeled_coordinates.extend(splitted_coordinates)

                # Ensure we have labeled coordinates before sampling
                if labeled_coordinates:
                    random_coors = random.sample(labeled_coordinates, 1)
                    self.train_coordinates.extend(random_coors)

            else:
                (start_d0, end_d0), (start_d1, end_d1), (start_d2, end_d2) = get_random_crop_coordinates(data_shape)
                self.train_coordinates.append(((start_d0, end_d0), (start_d1, end_d1), (start_d2, end_d2)))
    def __len__(self):
        return len(self.train_coordinates)
    def __getitem__(self, idx):
        (start_d0, end_d0), (start_d1, end_d1), (start_d2, end_d2) = self.train_coordinates[idx]
        image_block = self.normalized_image_data[start_d0:end_d0, start_d1:end_d1, start_d2:end_d2]
        label_block = self.label_data[start_d0:end_d0, start_d1:end_d1, start_d2:end_d2]
        return torch.tensor(image_block).unsqueeze(0).float(), torch.tensor(label_block).unsqueeze(0).float()
        # # Now `train_coordinates` contains the selected regions
        #     print((train_coordinates))
        #     plt.imshow(image_data[:, :, 53], cmap='gray')
        #     plt.title(f"Slice {slice_index}")
        #     plt.axis('off')  # Hide the axis
        #     plt.show()
dataset = AneurysmDataset('10055B')
loader = DataLoader(dataset, 2)
for idx, batch in enumerate(loader):
    image, label = batch
    print(image.size())
    print(label.size())
