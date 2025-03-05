##### import torch
import torch
from train import train_3d_unet
from unet_3d import UNet3D

import numpy as np
import os
train_list = ['10021', '10022', '10023', '10024', '10025', '10029', '10030', '10031', '10032', '10034', '10035', '10036', '10037', '10039', '10040', '10043', '10044B', '10044F', '10045B', '10047B', '10047F', '10048B', '10048F', '10049B', '10049F', '10050B', '10050F', '10051B', '10051F', '10052B', '10052F', '10053B', '10053F', '10054F', '10055B', '10055F', '10056B', '10056F', '10057B', '10057F', '10058B', '10058F', '10059B', '10059F', '10060B', '10060F', '10061B', '10061F', '10062B', '10062F', '10063B', '10063F', '10064B', '10064F', '10065B', '10065F', '10066F', '10067B', '10067F', '10068B', '10068F', '10069B', '10069F', '10070B', '10071B', '10071F', '10072B', '10072F', '10073B', '10074B', '10074F', '10075B', '10075F', '10076B', '10076F', '10077B', '10078B', '10078F']
valid_list = ['10026', '10027', '10028', '10054B', '10066B', '10070F', '10073F', '10077F', '10033', '10038', '10041', '10042', '10045F', '10046B', '10046F']
# Define the model
model = UNet3D(in_channels=1, num_classes=1)  # Replace with your actual 3D U-Net model class
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# model.load_state_dict(torch.load('/kaggle/input/checkpoint/best_256_adam_5325.pt'))
model = model.to(device)
model = torch.nn.DataParallel(model, device_ids = [0,1])

train_3d_unet(model, train_list, valid_list, 1000, device=device)