import torch
import pandas as pd
import numpy as np
import PIL
from PIL import Image
import math
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn.functional as F
import glob
import matplotlib.pyplot as plt

# With resize
class CustomDataset(Dataset):

    def __init__(self, csv_file, num_slices_3d = 7, extension='bmp', transform=False, image_size = 256,
                 mask_resize_ratio = 8, data_type = np.float32):
        self.csv_file = pd.read_csv(csv_file)
        self.transform = transform
        self.num_slices_3d = num_slices_3d
        self.extension = '/*.'+extension
        self.image_size = image_size
        self.data_type = data_type
        self.mask_resize_ratio = mask_resize_ratio
        # ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)
        # self.colorJitter = transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)
        self.colorJitter = transforms.ColorJitter(0.15, 0.15, 0.15, 0.15)
        self.tt = transforms.ToTensor()

    def __len__(self):
        return len(self.csv_file)

    def transform_basic(self, image, mask):
        '''

        :param image: PIL
        :param mask: PIL

        Performs the followings:
        functional.resize
        transforms.ToTensor

        :return: (image, mask)
        '''

        # resize, Dafault filter is PIL.Image.BILINEAR
        image = transforms.functional.resize(image, (self.image_size, self.image_size))
        mask = transforms.functional.resize(mask, (int(self.image_size/self.mask_resize_ratio),
                                                   int(self.image_size/self.mask_resize_ratio)))
        # To Tensor
        image = self.tt(image)
        # mask = self.tt(mask)
        mask = F.relu(self.tt(mask) - 0.1).sign()

        return image.unsqueeze(1), mask.unsqueeze(1)

    def transform_augmentation(self, image, mask):
        '''
        :param image: PIL
        :param mask: PIL

        Performs the followings:
        functional.crop
        transforms.functional.hflip
        torchvision.transforms.ColorJitter
        functional.affine
        functional.resize
        transforms.ToTensor

        :return: (image, mask)
        '''

# crop
        image = transforms.functional.crop(image, self.ratio_crop_top, self.ratio_crop_left, self.ratio_crop_size, self.ratio_crop_size)
        mask = transforms.functional.crop(mask, self.ratio_crop_top, self.ratio_crop_left, self.ratio_crop_size, self.ratio_crop_size)
# flip
        if self.ratio_RandomHorizontalFlip >= 0.5:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)
# ColorJitter
        image = self.colorJitter(image)
# affine
#         torchvision.transforms.functional.affine(img, angle, translate, scale, shear, resample=0, fillcolor=None)
        image = transforms.functional.affine(image, self.ratio_affine_angle, [0,0], 1.0, self.ratio_affine_shear, resample = PIL.Image.NEAREST)
        mask = transforms.functional.affine(mask, self.ratio_affine_angle, [0, 0], 1.0, self.ratio_affine_shear, resample = PIL.Image.NEAREST)
# resize, Dafault filter is PIL.Image.BILINEAR
        image = transforms.functional.resize(image, (self.image_size, self.image_size))
        mask = transforms.functional.resize(mask, (int(self.image_size/self.mask_resize_ratio),
                                                   int(self.image_size/self.mask_resize_ratio)))
# To Tensor
#         image = transforms.ToTensor(image)
#         mask = transforms.ToTensor(mask)
        image = self.tt(image)

        mask = F.relu(self.tt(mask) - 0.1).sign()
        # mask = self.tt(mask)
        return image.unsqueeze(1), mask.unsqueeze(1)

    def __getitem__(self, idx):
        # print("__getitem__!!!")
        if torch.is_tensor(idx):
            idx = idx.tolist()

        dir_image = sorted(glob.glob(self.csv_file.iloc[idx, 2]+self.extension))
        dir_mask = sorted(glob.glob(self.csv_file.iloc[idx, 3] + self.extension))

        # torch.nn.Conv3d : (N, C, D(3rd dim), H, W)

        if self.transform is False:
            batch_image, batch_mask = self.transform_basic(Image.open(dir_image[0]), Image.open(dir_mask[0]))

            # image with smaller slice #(idx_slice) -> closer to head
            for idx_slice in range(1, len(dir_image)):
                temp_set = self.transform_basic(Image.open(dir_image[idx_slice]), Image.open(dir_mask[idx_slice]))
                print("temp_set: "+str(len(temp_set)))
                print("temp_set[0]: " + str(temp_set[0].size()))
                print("temp_set[1]: " + str(temp_set[1].size()))
                batch_image = torch.cat([batch_image, temp_set[0]], dim=1)
                batch_mask = torch.cat([batch_mask, temp_set[1]], dim=1)


        elif self.transform is True:
            batch_image = Image.open(dir_image[0])
            batch_mask = Image.open(dir_mask[0])

            h, w = batch_image.size
            h = float(h)
            # if h != w:
            #     raise ValueError("image is not square!!!: "+str(idx))

            self.ratio_crop_top = int(round(0.1 * h * random.random()))
            self.ratio_crop_left = int(round(0.1 * h * random.random()))
            self.ratio_crop_size = int(round(h * random.uniform(0.9, 1.0)))
            self.ratio_RandomHorizontalFlip = random.random()
            self.ratio_affine_angle = random.uniform(-10.0, 10.0)
            self.ratio_affine_shear = random.uniform(-10.0, 10.0)

            batch_image, batch_mask = self.transform_augmentation(batch_image, batch_mask)

# image with smaller slice #(idx_slice) -> closer to head
            for idx_slice in range(1, len(dir_image)):
                temp_set = self.transform_augmentation(Image.open(dir_image[idx_slice]), Image.open(dir_mask[idx_slice]))
                batch_image = torch.cat([batch_image, temp_set[0]], dim=1)
                batch_mask = torch.cat([batch_mask, temp_set[1]], dim=1)


# size: [1, 7, 256, 256], [3, 7, 256, 256], [7], [7]
        # sample = {'image': batch_image, 'mask': batch_mask, 'class': self.csv_file.iloc[idx, 0]}
        # mask[0]: background, mask[1]: rectum + rectal cancer, mask[2]: rectal cancer
        sample = batch_image, batch_mask, float(self.csv_file.iloc[idx, 0]), self.csv_file.iloc[idx, 1]

        return sample












class CustomDataset_without_mask(Dataset):

    def __init__(self, csv_file, num_slices_3d = 7, extension='bmp', transform=False, image_size = 256,
                 mask_resize_ratio = 8, data_type = np.float32):
        self.csv_file = pd.read_csv(csv_file)
        self.transform = transform
        self.num_slices_3d = num_slices_3d
        self.extension = '/*.'+extension
        self.image_size = image_size
        self.data_type = data_type
        self.mask_resize_ratio = mask_resize_ratio
        # ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)
        # self.colorJitter = transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)
        self.colorJitter = transforms.ColorJitter(0.15, 0.15, 0.15, 0.15)
        self.tt = transforms.ToTensor()

    def __len__(self):
        return len(self.csv_file)

    def transform_basic(self, image):
        '''

        :param image: PIL
        :param mask: PIL

        Performs the followings:
        functional.resize
        transforms.ToTensor

        :return: (image, mask)
        '''

        # resize, Dafault filter is PIL.Image.BILINEAR
        image = transforms.functional.resize(image, (self.image_size, self.image_size))
        # To Tensor
        image = self.tt(image)

        return image.unsqueeze(1)

    def transform_augmentation(self, image):
        '''
        :param image: PIL
        :param mask: PIL

        Performs the followings:
        functional.crop
        transforms.functional.hflip
        torchvision.transforms.ColorJitter
        functional.affine
        functional.resize
        transforms.ToTensor

        :return: (image, mask)
        '''

# crop
        image = transforms.functional.crop(image, self.ratio_crop_top, self.ratio_crop_left, self.ratio_crop_size, self.ratio_crop_size)
# flip
        if self.ratio_RandomHorizontalFlip >= 0.5:
            image = transforms.functional.hflip(image)
# ColorJitter
        image = self.colorJitter(image)
# affine
#         torchvision.transforms.functional.affine(img, angle, translate, scale, shear, resample=0, fillcolor=None)
        image = transforms.functional.affine(image, self.ratio_affine_angle, [0,0], 1.0, self.ratio_affine_shear, resample = PIL.Image.NEAREST)
# resize, Dafault filter is PIL.Image.BILINEAR
        image = transforms.functional.resize(image, (self.image_size, self.image_size))
# To Tensor
#         image = transforms.ToTensor(image)
        image = self.tt(image)

        return image.unsqueeze(1)

    def __getitem__(self, idx):
        # print("__getitem__!!!")
        if torch.is_tensor(idx):
            idx = idx.tolist()

        dir_image = sorted(glob.glob(self.csv_file.iloc[idx, 2]+self.extension))

        # torch.nn.Conv3d : (N, C, D(3rd dim), H, W)

        if self.transform is False:
            batch_image = self.transform_basic(Image.open(dir_image[0]))

            # image with smaller slice #(idx_slice) -> closer to head
            for idx_slice in range(1, len(dir_image)):
                batch_image = torch.cat([batch_image, self.transform_basic(Image.open(dir_image[idx_slice]))], dim=1)

        elif self.transform is True:
            batch_image = Image.open(dir_image[0])

            h, w = batch_image.size
            h = float(h)
            # if h != w:
            #     raise ValueError("image is not square!!!: "+str(idx))

            self.ratio_crop_top = int(round(0.1 * h * random.random()))
            self.ratio_crop_left = int(round(0.1 * h * random.random()))
            self.ratio_crop_size = int(round(h * random.uniform(0.9, 1.0)))
            self.ratio_RandomHorizontalFlip = random.random()
            self.ratio_affine_angle = random.uniform(-10.0, 10.0)
            self.ratio_affine_shear = random.uniform(-10.0, 10.0)

            batch_image = self.transform_augmentation(batch_image)

# image with smaller slice #(idx_slice) -> closer to head
            for idx_slice in range(1, len(dir_image)):
                batch_image = torch.cat([batch_image, self.transform_augmentation(Image.open(dir_image[idx_slice]))], dim=1)


# size: [1, 7, 256, 256], [3, 7, 256, 256], [7], [7]
        # sample = {'image': batch_image, 'mask': batch_mask, 'class': self.csv_file.iloc[idx, 0]}
        # mask[0]: background, mask[1]: rectum + rectal cancer, mask[2]: rectal cancer
        sample = batch_image, float(self.csv_file.iloc[idx, 0]), self.csv_file.iloc[idx, 1]

        return sample


#
#
#
# csv_tr = 'D:/Rectum_exp/Data/data_path_3d/training_fold_1_ex.csv'
# csv_val = 'D:/Rectum_exp/Data/data_path_3d/validation_fold_1_ex.csv'
# # ds_tr = CustomDataset(csv_tr, transform=True)
# # ds_val = CustomDataset(csv_val, transform=False)
# ds_tr = CustomDataset_without_mask(csv_tr, transform=True)
# ds_val = CustomDataset_without_mask(csv_val, transform=False)
# for i in range(len(ds_tr)):
#     sample = ds_tr[i]
#
#     print(i)
#     # print(sample['image'].size(), sample['mask'].size(), sample['class'])
#     # print(sample[0].size(), sample[1].size(), sample[2], sample[3])
#     print(sample[0].size(), sample[1], sample[2])
#     fig = plt.figure()
#     # fig.add_subplot(1, 3, 1)
#     plt.imshow(sample[0][0, 3, :, :], cmap='Greys_r')
#     # fig.add_subplot(1, 3, 2)
#     # plt.imshow(sample[1][1, 3, :, :], cmap='Greys_r')
#     # fig.add_subplot(1, 3, 3)
#     # plt.imshow(sample[1][2, 3, :, :], cmap='Greys_r')
#     # plt.title(str(int(sample[2]))+" / "+str(sample[3]))
#     plt.show()
