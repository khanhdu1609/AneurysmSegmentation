import numpy as np
import matplotlib.pyplot as plt

# Load the 3D image data (x, y, z)
image_data = np.load("C:/Users/XgearHN_Lap/Downloads/pre_aneurysm_dataset/10055B/image.npy")

# Print shape to understand the dimensions
print("Shape of image data (x, y, z):", image_data.shape)

# Select the middle slice along the z-axis
z_middle = image_data.shape[2] // 2

# Plot the selected slice
plt.figure(figsize=(6, 6))
plt.imshow(image_data[:, :, z_middle], cmap="gray")  # Change cmap if needed
plt.colorbar()
plt.title(f"Middle Slice at Z={z_middle}")
plt.show()
