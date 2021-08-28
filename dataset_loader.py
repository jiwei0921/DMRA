import os

import numpy as np
import PIL.Image
import scipy.io as sio
import torch
from torch.utils import data
import cv2

class MyData(data.Dataset):  # inherit
    """
    load data in a folder
    """
    mean_rgb = np.array([0.447, 0.407, 0.386])
    std_rgb = np.array([0.244, 0.250, 0.253])
    def __init__(self, root, transform=False):
        super(MyData, self).__init__()
        self.root = root

        self._transform = transform

        img_root = os.path.join(self.root, 'train_images')
        lbl_root = os.path.join(self.root, 'train_masks')
        depth_root = os.path.join(self.root, 'train_depth')

        file_names = os.listdir(img_root)
        self.img_names = []
        self.lbl_names = []
        self.depth_names = []
        for i, name in enumerate(file_names):
            if not name.endswith('.jpg'):
                continue
            self.lbl_names.append(
                os.path.join(lbl_root, name[:-4]+'.png')
            )
            self.img_names.append(
                os.path.join(img_root, name)
            )
            self.depth_names.append(
                os.path.join(depth_root, name[:-4]+'.png')
            )

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        # load image
        img_file = self.img_names[index]
        img = PIL.Image.open(img_file)
        img = np.array(img, dtype=np.uint8)
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        # load label
        lbl_file = self.lbl_names[index]
        lbl = PIL.Image.open(lbl_file).convert('L')
        lbl = np.array(lbl, dtype=np.int32)
        lbl = cv2.resize(lbl, (256, 256), interpolation=cv2.INTER_AREA)
        lbl[lbl != 0] = 1
        # load depth
        depth_file = self.depth_names[index]
        depth = PIL.Image.open(depth_file).convert('L')
        depth = np.array(depth, dtype=np.uint8)
        depth = cv2.resize(depth, (256, 256), interpolation=cv2.INTER_AREA)



        if self._transform:
            return self.transform(img, lbl, depth)
        else:
            return img, lbl, depth


    # Translating numpy_array into format that pytorch can use on Code.
    def transform(self, img, lbl, depth):

        img = img.astype(np.float64)/255.0
        img -= self.mean_rgb
        img /= self.std_rgb
        img = img.transpose(2, 0, 1)  # to verify
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        depth = depth.astype(np.float64)/255.0
        depth = torch.from_numpy(depth).float()
        return img, lbl, depth


class MyTestData(data.Dataset):
    """
    load data in a folder
    """
    mean_rgb = np.array([0.447, 0.407, 0.386])
    std_rgb = np.array([0.244, 0.250, 0.253])


    def __init__(self, root, transform=False):
        super(MyTestData, self).__init__()
        self.root = root
        self._transform = transform

        img_root = os.path.join(self.root, 'test_images')
        depth_root = os.path.join(self.root, 'test_depth')
        file_names = os.listdir(img_root)
        self.img_names = []
        self.names = []
        self.depth_names = []

        for i, name in enumerate(file_names):
            if not name.endswith('.jpg'):
                continue
            self.img_names.append(
                os.path.join(img_root, name)
            )
            self.names.append(name[:-4])
            self.depth_names.append(
                # os.path.join(depth_root, name[:-4]+'_depth.png')        # Test RGBD135 dataset
                os.path.join(depth_root, name[:-4] + '.png')
            )

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        # load image
        img_file = self.img_names[index]
        img = PIL.Image.open(img_file)
        img_size = img.size
        img = np.array(img, dtype=np.uint8)
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)

        # load focal
        depth_file = self.depth_names[index]
        depth = PIL.Image.open(depth_file).convert('L')
        depth = np.array(depth, dtype=np.uint8)
        depth = cv2.resize(depth, (256, 256), interpolation=cv2.INTER_AREA)
        if self._transform:
            img, focal = self.transform(img, depth)
            return img, focal, self.names[index], img_size
        else:
            return img, depth, self.names[index], img_size

    def transform(self, img, depth):
        img = img.astype(np.float64)/255.0
        img -= self.mean_rgb
        img /= self.std_rgb
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()

        depth = depth.astype(np.float64)/255.0
        depth = torch.from_numpy(depth).float()

        return img, depth
