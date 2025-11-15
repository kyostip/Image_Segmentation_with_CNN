from torch.utils.data import Dataset
import cv2
import numpy as np


def read_xray(path):
    xray = cv2.imread(path, cv2.IMREAD_GRAYSCALE) # H * W
    if xray is None:
        raise FileNotFoundError(f"Could not read {path}")
    xray = xray.astype(np.float32) / 255.0
    # note!  
    xray = xray.reshape((1, *xray.shape)) # 1.h.w
    return xray


def read_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # note!
    #note !
    mask = (mask > 0 ).astype(np.float32)
    mask = mask.reshape((1, *mask.shape)) # 1.h.w
    return mask

class xray_dataset(Dataset):
    def __init__(self, df):
        self.df = df


    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image = read_xray(self.df['xrays'].iloc[index])
        mask = read_mask(self.df['masks'].iloc[index])


        results = {
            'image': image,
            'mask': mask
        }
        return results


class test_dataset(Dataset):
    def __init__(self, df):
        self.df = df


    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image = read_xray(self.df['xrays'].iloc[index])



        results = {
            'image': image,
        }
        return results