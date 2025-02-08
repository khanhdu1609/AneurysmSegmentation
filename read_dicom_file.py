import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from pydicom import dcmread
import glob
import math
def read_image(dicom_folder):
    path_files = glob.glob(dicom_folder + "/*.dcm")
    slices = [pydicom.read_file(s) for s in path_files]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    if slices[0].ImagePositionPatient[2] == slices[1].ImagePositionPatient[2]:
        sec_num = 2;
        while slices[0].ImagePositionPatient[2] == slices[sec_num].ImagePositionPatient[2]:
            sec_num = sec_num+1;
        slice_num = int(len(slices) / sec_num)
        slices.sort(key = lambda x:float(x.InstanceNumber))
        slices = slices[0:slice_num]
        slices.sort(key = lambda x:float(x.ImagePositionPatient[2]))

    # Pixel processing
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)

    # # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

        # window width and window level
        ww = slices[slice_number].WindowWidth
        wl = slices[slice_number].WindowCenter
        if not isinstance(ww, pydicom.valuerep.DSfloat):
            ww = ww[0]
        if not isinstance(wl, pydicom.valuerep.DSfloat):
            wl = wl[0]

        w_min, w_max = wl - ww // 2, wl + ww // 2
        image[slice_number][image[slice_number] < w_min] = w_min
        image[slice_number][image[slice_number] > w_max] = w_max
        image[slice_number] = ((1.0 * (image[slice_number] - w_min) / (w_max - w_min)) * 255).astype(np.uint8)

    return np.array(image, dtype=np.uint8)

def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    if not isinstance(mask_rle, str):
      return np.zeros(shape[0]*shape[1]*shape[2], dtype=np.uint8).reshape(shape)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1]*shape[2], dtype=np.uint8)
    
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

def read_label(path_csv, h, w, num_slice):
    annot_data = pd.read_csv(path_csv, index_col=0)
    labels = np.zeros((num_slice, 1, h, w), dtype=np.uint8)

    for index_s in range(num_slice):
        if (annot_data["RLE"][index_s] is np.NAN):
            continue
        labels[index_s, :, :, :] = rle_decode(annot_data["RLE"][index_s],
                                                   (1, h, w))
    return labels[:,0,:,:]
dicom = dcmread('Ca Van Phung/CA VAN PHUNG 1965.Seq5.Ser9.Img426.dcm')
image = np.asarray(dicom.pixel_array)
print(np.min(image))
print(np.max(image))