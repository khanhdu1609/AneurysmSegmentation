import numpy as np
import os
root_folder = "C:/Users/XgearHN_Lap/Downloads/pre_aneurysm_dataset"
for subfolder in os.listdir(root_folder):
    subfolder_path = os.path.join(root_folder, subfolder)
    image_path = os.path.join(subfolder_path, 'image.npy')
    image = np.load(image_path)
    image = image.transpose(0, 2, 1)
    np.save(image_path, image)

    label_path = os.path.join(subfolder_path, 'label.npy')
    label = np.load(label_path)
    label = label.transpose(0, 2, 1)
    np.save(label_path, label)