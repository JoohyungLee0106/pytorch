import torch
import pandas as pd
import numpy as np
from PIL import Image
import PIL
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn.functional as F
import glob
import os
import matplotlib.pyplot as plt


# CustomDataset_3D_no_mask_center_crop
# CustomDataset_3D_no_mask

# CustomDataset_2D_no_mask_center_crop
# CustomDataset_2D_no_mask


class CustomDataset_3D_no_mask_center_crop(Dataset):

    def __init__(self, csv_file, num_slices_3d = 7, extension='bmp', transform=False, image_size = 256,
                 mask_resize_ratio = 1.0, label_threshold=0.1, channel_for_cropping = 1, data_type = np.float32):

        self.csv_file = pd.read_csv(csv_file)
        self.num_slices_3d = num_slices_3d
        self.extension = '/*.'+extension
        self.image_size = image_size
        self.data_type = data_type
        self.mask_resize_ratio = mask_resize_ratio
        self.channel_for_cropping = channel_for_cropping
        # ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)
        # self.colorJitter = transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)
        self.colorJitter = transforms.ColorJitter(0.2, 0.2, 0.2, 0.2)
        self.tt = transforms.ToTensor()
        self.label_threshold = label_threshold
        self.batch_image = torch.zeros((1, self.num_slices_3d, self.image_size, self.image_size))
        self.batch_mask = torch.zeros((3, self.num_slices_3d, self.image_size, self.image_size))

        if transform:
            self.transform = self.processing_training
        else:
            self.transform = self.processing_validation

    def __len__(self):
        return len(self.csv_file)

    def get_cropping_boundary(self, mask):
        '''
        self.cropping_boundary: [top, bottom, left, right]
        :param mask: PIL,  [H, W, D]
        '''
        __mask_np_temp = np.copy(np.array(mask)[:, :, 1])
        # __mask_np_temp[__mask_np_temp > self.label_threshold] = 1
        __mask_np_temp[__mask_np_temp <= self.label_threshold] = 0
        __mask_np_temp = np.nonzero(__mask_np_temp)

        return [__mask_np_temp[0].min(), __mask_np_temp[0].max(), __mask_np_temp[1].min(), __mask_np_temp[1].max()]

    def transform_basic(self, image, resize_ratio):
        '''

        :param image: PIL
        :param resize_ratio: int or float
        :return:
        '''

        # resize, Dafault filter is PIL.Image.BILINEAR
        image = transforms.functional.resize(image, (int(self.image_size/resize_ratio),
                                                   int(self.image_size/resize_ratio)))
        # To Tensor
        image = self.tt(image)
        # mask = self.tt(mask)
        # mask = F.relu(self.tt(image) - self.label_threshold).sign()

        return image.unsqueeze(1)

    def processing_validation(self, path_images_list, path_masks_list):
    # def processing_validation(self, path_images_list, path_masks_list):
        '''
        :param path_images: sorted
        :param path_masks: sorted
        :return:
        '''

        # C D H W
        for idx_slice in range(len(path_images_list)):
            self.batch_image[:, idx_slice, :, :] = self.transform_basic(Image.open(path_images_list[idx_slice]), 1)
            # self.batch_mask[:, idx_slice, :, :] = F.relu(self.transform_basic(Image.open(path_masks_list[idx_slice]), 1)
            #                                              - self.label_threshold).sign()

    def processing_training(self, path_images_list, path_masks_list):
        '''
        :param path_images_list: list of image path
        :param path_masks_list: list of mask path
        :return:
        '''
        # print(path_images_list)
        ### coefficients
        h, w = (Image.open(path_images_list[0])).size
        coef_RandomHorizontalFlip = random.random()
        coef_affine_angle = random.uniform(-10.0, 10.0)
        coef_affine_shear = random.uniform(-10.0, 10.0)
        __top = [0] * self.num_slices_3d
        __bottom = [0] * self.num_slices_3d
        __left = [0] * self.num_slices_3d
        __right = [0] * self.num_slices_3d
        ###

        __image_temp_list = [0] * self.num_slices_3d
        __mask_temp_list = [0] * self.num_slices_3d

        # batch_image_temp = torch.zeros((1, self.num_slices_3d, h, w))
        # batch_mask_temp = torch.zeros((3, self.num_slices_3d, h, w))
        batch_mask_np = np.zeros((self.num_slices_3d, h, w))

        # C D H W
        for idx_slice in range(len(path_images_list)):
            __image_temp_list[idx_slice] = Image.open(path_images_list[idx_slice])
            __mask_temp_list[idx_slice] = Image.open(path_masks_list[idx_slice])

            if coef_RandomHorizontalFlip > 0.5:
                __image_temp_list[idx_slice] = transforms.functional.hflip(__image_temp_list[idx_slice])
                __mask_temp_list[idx_slice] = transforms.functional.hflip(__mask_temp_list[idx_slice])

            __image_temp_list[idx_slice] = transforms.functional.affine(__image_temp_list[idx_slice], coef_affine_angle, [0, 0], 1.0, coef_affine_shear,
                                             resample=PIL.Image.NEAREST)
            __mask_temp_list[idx_slice] = transforms.functional.affine(__mask_temp_list[idx_slice], coef_affine_angle, [0, 0], 1.0, coef_affine_shear,
                                             resample=PIL.Image.NEAREST)

            __top[idx_slice], __bottom[idx_slice], __left[idx_slice], __right[idx_slice] =\
                self.get_cropping_boundary(__mask_temp_list[idx_slice])

        #     ##########
        #     batch_mask_np[idx_slice, :, :] = np.copy(np.array(__mask_temp_list[idx_slice])[:,:,1])
        #     ##########
        # batch_mask_np[batch_mask_np <= self.label_threshold] = 0
        # np.sum(batch_mask_np, axis=0, keepdims=False)
        # merged_label = np.nonzero(batch_mask_np)
        # __top2 = merged_label[0].min()
        # __bottom2 = merged_label[0].max()
        # __left2 = merged_label[1].min()
        # __right2 = merged_label[1].max()

        __top_limit = int(min(__top) * 0.8)
        __bottom_limit = max(__bottom) + int((h-max(__bottom))*0.2)
        __left_limit = int(min(__left)*0.7)
        __right_limit = max(__right) + int((w-max(__right))*0.3)
        # __top = min(__top)
        # __bottom = max(__bottom)
        # __left = min(__left)
        # __right = max(__right)
        __center = int(h/2.0)
        __cropping_size_min = max( [__center-__top_limit, __bottom_limit-__center, __center-__left_limit, __right_limit-__center] )*2
        __cropping_size_coeff = random.random()
        __cropping_size = (__cropping_size_coeff*__cropping_size_min)+((1-__cropping_size_coeff)*h)
        # if_height_was_found = False
        # # cnt = 0
        # while not(if_height_was_found):
        #
        #     __top_to_crop = round(__top_limit * random.random())
        #     __left_to_crop = round(__left_limit * random.random())
        #
        #     __cropping_size_min = max([(__bottom_limit-__top_to_crop), (__right_limit-__left_to_crop)])
        #     __cropping_size_max = min([(h - __top_to_crop), (w - __left_to_crop)])
        #     if (__cropping_size_max >= __cropping_size_min):
        #         __cropping_size_coeff = random.random()
        #         __cropping_size = (__cropping_size_coeff*__cropping_size_min) + ((1-__cropping_size_coeff)*__cropping_size_max)
        #         if_height_was_found = True
        #     # else:
        #     #     cnt+=1
        #     #     print("# that height was not found: "+str(cnt))





        # No more mask
        for idx_slice in range(len(path_images_list)):
            # __image_temp_list[idx_slice] = transforms.functional.crop(__image_temp_list[idx_slice],
            #                                             __top_to_crop, __left_to_crop, __cropping_size, __cropping_size)
            # __image_temp_list[idx_slice] = transforms.functional.center_crop(__image_temp_list[idx_slice], (__cropping_size*2))
            __image_temp_list[idx_slice] = transforms.functional.center_crop(__image_temp_list[idx_slice], __cropping_size)

            # resize, Dafault filter is PIL.Image.BILINEAR
            __image_temp_list[idx_slice] = transforms.functional.resize(__image_temp_list[idx_slice], (self.image_size, self.image_size))

            __image_temp_list[idx_slice] = self.colorJitter(__image_temp_list[idx_slice])

            __image_temp_list[idx_slice] = self.tt(__image_temp_list[idx_slice])

            __image_temp_list[idx_slice].unsqueeze(1)

            self.batch_image[:, idx_slice, :, :] = __image_temp_list[idx_slice]

    def processing_data_id(self, idx):
        '''
        For 3d data
        :param idx:
        :return:
        '''
        # 2d data
        # return os.path.split(self.csv_file.iloc[idx, 1])[-1]
        return self.csv_file.iloc[idx, 1]


    def __getitem__(self, idx):
        # print("__getitem__!!!")
        if torch.is_tensor(idx):
            idx = idx.tolist()

        path_images_list = sorted(glob.glob(self.csv_file.iloc[idx, 2] + self.extension))
        path_masks_list = sorted(glob.glob(self.csv_file.iloc[idx, 3] + self.extension))

        self.transform(path_images_list, path_masks_list)

        # size: [1, 7, 256, 256], [3, 7, 256, 256], [7], [7]
        # sample = {'image': batch_image, 'mask': batch_mask, 'class': self.csv_file.iloc[idx, 0]}
        # mask[0]: background, mask[1]: rectum + rectal cancer, mask[2]: rectal cancer
        # image, classification label, patient number (3D)
        return self.batch_image, float(self.csv_file.iloc[idx, 0]), self.processing_data_id(idx)


class CustomDataset_3D_no_mask(Dataset):

    def __init__(self, csv_file, num_slices_3d = 7, extension='bmp', transform=False, image_size = 256,
                 mask_resize_ratio = 1.0, label_threshold=0.1, channel_for_cropping = 1, data_type = np.float32):

        self.csv_file = pd.read_csv(csv_file)
        self.num_slices_3d = num_slices_3d
        self.extension = '/*.'+extension
        self.image_size = image_size
        self.data_type = data_type
        self.mask_resize_ratio = mask_resize_ratio
        self.channel_for_cropping = channel_for_cropping
        # ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)
        # self.colorJitter = transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)
        self.colorJitter = transforms.ColorJitter(0.2, 0.2, 0.2, 0.2)
        self.tt = transforms.ToTensor()
        self.label_threshold = label_threshold
        self.batch_image = torch.zeros((1, self.num_slices_3d, self.image_size, self.image_size))
        self.batch_mask = torch.zeros((3, self.num_slices_3d, self.image_size, self.image_size))

        if transform:
            self.transform = self.processing_training
        else:
            self.transform = self.processing_validation

    def __len__(self):
        return len(self.csv_file)

    def get_cropping_boundary(self, mask):
        '''
        self.cropping_boundary: [top, bottom, left, right]
        :param mask: PIL,  [H, W, D]
        '''
        __mask_np_temp = np.copy(np.array(mask)[:, :, 1])
        # __mask_np_temp[__mask_np_temp > self.label_threshold] = 1
        __mask_np_temp[__mask_np_temp <= self.label_threshold] = 0
        __mask_np_temp = np.nonzero(__mask_np_temp)

        return [__mask_np_temp[0].min(), __mask_np_temp[0].max(), __mask_np_temp[1].min(), __mask_np_temp[1].max()]

    def transform_basic(self, image, resize_ratio):
        '''

        :param image: PIL
        :param resize_ratio: int or float
        :return:
        '''

        # resize, Dafault filter is PIL.Image.BILINEAR
        image = transforms.functional.resize(image, (int(self.image_size/resize_ratio),
                                                   int(self.image_size/resize_ratio)))
        # To Tensor
        image = self.tt(image)
        # mask = self.tt(mask)
        # mask = F.relu(self.tt(image) - self.label_threshold).sign()

        return image.unsqueeze(1)

    def processing_validation(self, path_images_list, path_masks_list):
    # def processing_validation(self, path_images_list, path_masks_list):
        '''
        :param path_images: sorted
        :param path_masks: sorted
        :return:
        '''

        # C D H W
        for idx_slice in range(len(path_images_list)):
            self.batch_image[:, idx_slice, :, :] = self.transform_basic(Image.open(path_images_list[idx_slice]), 1)
            # self.batch_mask[:, idx_slice, :, :] = F.relu(self.transform_basic(Image.open(path_masks_list[idx_slice]), 1)
            #                                              - self.label_threshold).sign()

    def processing_training(self, path_images_list, path_masks_list):
        '''
        :param path_images_list: list of image path
        :param path_masks_list: list of mask path
        :return:
        '''
        # print(path_images_list)
        ### coefficients
        h, w = (Image.open(path_images_list[0])).size
        coef_RandomHorizontalFlip = random.random()
        coef_affine_angle = random.uniform(-10.0, 10.0)
        coef_affine_shear = random.uniform(-10.0, 10.0)
        __top = [0] * self.num_slices_3d
        __bottom = [0] * self.num_slices_3d
        __left = [0] * self.num_slices_3d
        __right = [0] * self.num_slices_3d
        ###

        __image_temp_list = [0] * self.num_slices_3d
        __mask_temp_list = [0] * self.num_slices_3d

        # batch_image_temp = torch.zeros((1, self.num_slices_3d, h, w))
        # batch_mask_temp = torch.zeros((3, self.num_slices_3d, h, w))
        batch_mask_np = np.zeros((self.num_slices_3d, h, w))

        # C D H W
        for idx_slice in range(len(path_images_list)):
            __image_temp_list[idx_slice] = Image.open(path_images_list[idx_slice])
            __mask_temp_list[idx_slice] = Image.open(path_masks_list[idx_slice])

            if coef_RandomHorizontalFlip > 0.5:
                __image_temp_list[idx_slice] = transforms.functional.hflip(__image_temp_list[idx_slice])
                __mask_temp_list[idx_slice] = transforms.functional.hflip(__mask_temp_list[idx_slice])

            __image_temp_list[idx_slice] = transforms.functional.affine(__image_temp_list[idx_slice], coef_affine_angle, [0, 0], 1.0, coef_affine_shear,
                                             resample=PIL.Image.NEAREST)
            __mask_temp_list[idx_slice] = transforms.functional.affine(__mask_temp_list[idx_slice], coef_affine_angle, [0, 0], 1.0, coef_affine_shear,
                                             resample=PIL.Image.NEAREST)

            __top[idx_slice], __bottom[idx_slice], __left[idx_slice], __right[idx_slice] =\
                self.get_cropping_boundary(__mask_temp_list[idx_slice])

        #     ##########
        #     batch_mask_np[idx_slice, :, :] = np.copy(np.array(__mask_temp_list[idx_slice])[:,:,1])
        #     ##########
        # batch_mask_np[batch_mask_np <= self.label_threshold] = 0
        # np.sum(batch_mask_np, axis=0, keepdims=False)
        # merged_label = np.nonzero(batch_mask_np)
        # __top2 = merged_label[0].min()
        # __bottom2 = merged_label[0].max()
        # __left2 = merged_label[1].min()
        # __right2 = merged_label[1].max()

        __top_limit = int(min(__top) * 0.8)
        __bottom_limit = max(__bottom) + int((h-max(__bottom))*0.2)
        __left_limit = int(min(__left)*0.7)
        __right_limit = max(__right) + int((w-max(__right))*0.3)
        # __top = min(__top)
        # __bottom = max(__bottom)
        # __left = min(__left)
        # __right = max(__right)
        if_height_was_found = False
        # cnt = 0
        while not(if_height_was_found):

            __top_to_crop = round(__top_limit * random.random())
            __left_to_crop = round(__left_limit * random.random())

            __cropping_size_min = max([(__bottom_limit-__top_to_crop), (__right_limit-__left_to_crop)])
            __cropping_size_max = min([(h - __top_to_crop), (w - __left_to_crop)])
            if (__cropping_size_max >= __cropping_size_min):
                __cropping_size_coeff = random.random()
                __cropping_size = (__cropping_size_coeff*__cropping_size_min) + ((1-__cropping_size_coeff)*__cropping_size_max)
                if_height_was_found = True
            # else:
            #     cnt+=1
            #     print("# that height was not found: "+str(cnt))

        # No more mask
        for idx_slice in range(len(path_images_list)):
            __image_temp_list[idx_slice] = transforms.functional.crop(__image_temp_list[idx_slice],
                                                        __top_to_crop, __left_to_crop, __cropping_size, __cropping_size)

            # resize, Dafault filter is PIL.Image.BILINEAR
            __image_temp_list[idx_slice] = transforms.functional.resize(__image_temp_list[idx_slice], (self.image_size, self.image_size))

            __image_temp_list[idx_slice] = self.colorJitter(__image_temp_list[idx_slice])

            __image_temp_list[idx_slice] = self.tt(__image_temp_list[idx_slice])

            __image_temp_list[idx_slice].unsqueeze(1)

            self.batch_image[:, idx_slice, :, :] = __image_temp_list[idx_slice]

    def processing_data_id(self, idx):
        '''
        For 3d data
        :param idx:
        :return:
        '''
        # 2d data
        # return os.path.split(self.csv_file.iloc[idx, 1])[-1]
        return self.csv_file.iloc[idx, 1]


    def __getitem__(self, idx):
        # print("__getitem__!!!")
        if torch.is_tensor(idx):
            idx = idx.tolist()

        path_images_list = sorted(glob.glob(self.csv_file.iloc[idx, 2] + self.extension))
        path_masks_list = sorted(glob.glob(self.csv_file.iloc[idx, 3] + self.extension))

        self.transform(path_images_list, path_masks_list)

        # size: [1, 7, 256, 256], [3, 7, 256, 256], [7], [7]
        # sample = {'image': batch_image, 'mask': batch_mask, 'class': self.csv_file.iloc[idx, 0]}
        # mask[0]: background, mask[1]: rectum + rectal cancer, mask[2]: rectal cancer
        # image, classification label, patient number (3D)
        return self.batch_image, float(self.csv_file.iloc[idx, 0]), self.processing_data_id(idx)


#########################################################################



class CustomDataset_2D_no_mask_center_crop(Dataset):

    def __init__(self, csv_file, num_slices_3d = 7, extension='bmp', transform=False, image_size = 256,
                 mask_resize_ratio = 1.0, label_threshold=0.1, channel_for_cropping = 1, data_type = np.float32):

        self.csv_file = pd.read_csv(csv_file)
        self.image_size = image_size
        self.data_type = data_type
        self.mask_resize_ratio = mask_resize_ratio
        self.channel_for_cropping = channel_for_cropping
        # ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)
        # self.colorJitter = transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)
        self.colorJitter = transforms.ColorJitter(0.2, 0.2, 0.2, 0.2)
        self.tt = transforms.ToTensor()
        self.label_threshold = label_threshold

        if transform:
            self.transform = self.processing_training
        else:
            self.transform = self.processing_validation

    def __len__(self):
        return len(self.csv_file)

    def get_cropping_boundary(self, mask):
        '''
        self.cropping_boundary: [top, bottom, left, right]
        :param mask: PIL,  [H, W, C]
        '''
        __mask_np_temp = np.copy(np.array(mask)[:, :, 1])
        # __mask_np_temp[__mask_np_temp > self.label_threshold] = 1
        __mask_np_temp[__mask_np_temp <= self.label_threshold] = 0
        __mask_np_temp = np.nonzero(__mask_np_temp)

        return [__mask_np_temp[0].min(), __mask_np_temp[0].max(), __mask_np_temp[1].min(), __mask_np_temp[1].max()]

    def transform_basic(self, image, resize_ratio):
        '''

        :param image: PIL
        :param resize_ratio: int or float
        :return:
        '''

        # resize, Dafault filter is PIL.Image.BILINEAR
        image = transforms.functional.resize(image, (int(self.image_size/resize_ratio),
                                                   int(self.image_size/resize_ratio)))
        # To Tensor
        image = self.tt(image)
        # mask = self.tt(mask)
        # mask = F.relu(self.tt(image) - self.label_threshold).sign()

        return image

    def processing_validation(self, path_images, path_masks):
    # def processing_validation(self, path_images_list, path_masks_list):
        '''
        :param path_images: sorted
        :param path_masks: sorted
        :return:
        '''

        # C H W
        self.batch_image = self.transform_basic(Image.open(path_images), 1)
        # self.batch_mask[:, :, :] = F.relu(self.transform_basic(Image.open(path_masks_list[idx_slice]), 1)
        #                                              - self.label_threshold).sign()

    def processing_training(self, path_images, path_masks):
        '''
        :param path_images_list: list of image path
        :param path_masks_list: list of mask path
        :return:
        '''
        # C D H W
        __image_temp = Image.open(path_images)
        __mask_temp = Image.open(path_masks)
        # print(path_images_list)
        ### coefficients
        h, w = __image_temp.size
        coef_RandomHorizontalFlip = random.random()
        coef_affine_angle = random.uniform(-10.0, 10.0)
        coef_affine_shear = random.uniform(-10.0, 10.0)
        ###

        # batch_image_temp = torch.zeros((1, self.num_slices_3d, h, w))
        # batch_mask_temp = torch.zeros((3, self.num_slices_3d, h, w))

        if coef_RandomHorizontalFlip > 0.5:
            __image_temp = transforms.functional.hflip(__image_temp)
            __mask_temp = transforms.functional.hflip(__mask_temp)

        __image_temp = transforms.functional.affine(__image_temp, coef_affine_angle, [0, 0], 1.0, coef_affine_shear,
                                         resample=PIL.Image.NEAREST)
        __mask_temp = transforms.functional.affine(__mask_temp, coef_affine_angle, [0, 0], 1.0, coef_affine_shear,
                                         resample=PIL.Image.NEAREST)

        __top, __bottom, __left, __right = self.get_cropping_boundary(__mask_temp)

        #     ##########
        #     batch_mask_np[idx_slice, :, :] = np.copy(np.array(__mask_temp_list[idx_slice])[:,:,1])
        #     ##########
        # batch_mask_np[batch_mask_np <= self.label_threshold] = 0
        # np.sum(batch_mask_np, axis=0, keepdims=False)
        # merged_label = np.nonzero(batch_mask_np)
        # __top2 = merged_label[0].min()
        # __bottom2 = merged_label[0].max()
        # __left2 = merged_label[1].min()
        # __right2 = merged_label[1].max()

        __top_limit = int(__top * 0.8)
        __bottom_limit = __bottom + int((h-__bottom)*0.2)
        __left_limit = int(__left*0.7)
        __right_limit = __right + int((w-__right)*0.3)
        # __top = min(__top)
        # __bottom = max(__bottom)
        # __left = min(__left)
        # __right = max(__right)
        __center = int(h/2.0)
        __cropping_size_min = max( [__center-__top_limit, __bottom_limit-__center, __center-__left_limit, __right_limit-__center] )*2
        __cropping_size_coeff = random.random()
        __cropping_size = (__cropping_size_coeff*__cropping_size_min)+((1-__cropping_size_coeff)*h)

        # __image_temp_list[idx_slice] = transforms.functional.crop(__image_temp_list[idx_slice],
        #                                             __top_to_crop, __left_to_crop, __cropping_size, __cropping_size)
        # __image_temp_list[idx_slice] = transforms.functional.center_crop(__image_temp_list[idx_slice], (__cropping_size*2))
        __image_temp = transforms.functional.center_crop(__image_temp, __cropping_size)

        # resize, Dafault filter is PIL.Image.BILINEAR
        __image_temp = transforms.functional.resize(__image_temp, (self.image_size, self.image_size))

        __image_temp = self.colorJitter(__image_temp)

        __image_temp = self.tt(__image_temp)

        __image_temp.unsqueeze(1)

        self.batch_image = __image_temp

    def processing_data_id(self, idx):
        '''
        For 3d data
        :param idx:
        :return:
        '''
        # 2d data
        return os.path.split(self.csv_file.iloc[idx, 2])[-1]
        # return self.csv_file.iloc[idx, 1]


    def __getitem__(self, idx):
        # print("__getitem__!!!")
        if torch.is_tensor(idx):
            idx = idx.tolist()

        self.transform(self.csv_file.iloc[idx, 2], self.csv_file.iloc[idx, 3])

        # size: [1, 256, 256], [3,  256, 256], [7], [7]
        # sample = {'image': batch_image, 'mask': batch_mask, 'class': self.csv_file.iloc[idx, 0]}
        # mask[0]: background, mask[1]: rectum + rectal cancer, mask[2]: rectal cancer
        # image, classification label, patient number (2D)
        # NCHW
        return self.batch_image, float(self.csv_file.iloc[idx, 0]), self.processing_data_id(idx)







class CustomDataset_2D_no_mask(Dataset):

    def __init__(self, csv_file, num_slices_3d = 7, extension='bmp', transform=False, image_size = 256,
                 mask_resize_ratio = 1.0, label_threshold=0.1, channel_for_cropping = 1, data_type = np.float32):

        self.csv_file = pd.read_csv(csv_file)
        self.image_size = image_size
        self.data_type = data_type
        self.mask_resize_ratio = mask_resize_ratio
        self.channel_for_cropping = channel_for_cropping
        # ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)
        # self.colorJitter = transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)
        self.colorJitter = transforms.ColorJitter(0.2, 0.2, 0.2, 0.2)
        self.tt = transforms.ToTensor()
        self.label_threshold = label_threshold

        if transform:
            self.transform = self.processing_training
        else:
            self.transform = self.processing_validation

    def __len__(self):
        return len(self.csv_file)

    def get_cropping_boundary(self, mask):
        '''
        self.cropping_boundary: [top, bottom, left, right]
        :param mask: PIL,  [H, W, C]
        '''
        __mask_np_temp = np.copy(np.array(mask)[:, :, 1])
        # __mask_np_temp[__mask_np_temp > self.label_threshold] = 1
        __mask_np_temp[__mask_np_temp <= self.label_threshold] = 0
        __mask_np_temp = np.nonzero(__mask_np_temp)

        return [__mask_np_temp[0].min(), __mask_np_temp[0].max(), __mask_np_temp[1].min(), __mask_np_temp[1].max()]

    def transform_basic(self, image, resize_ratio):
        '''

        :param image: PIL
        :param resize_ratio: int or float
        :return:
        '''

        # resize, Dafault filter is PIL.Image.BILINEAR
        image = transforms.functional.resize(image, (int(self.image_size/resize_ratio),
                                                   int(self.image_size/resize_ratio)))
        # To Tensor
        image = self.tt(image)
        # mask = self.tt(mask)
        # mask = F.relu(self.tt(image) - self.label_threshold).sign()

        return image

    def processing_validation(self, path_images, path_masks):
    # def processing_validation(self, path_images_list, path_masks_list):
        '''
        :param path_images: sorted
        :param path_masks: sorted
        :return:
        '''

        # C H W
        self.batch_image = self.transform_basic(Image.open(path_images), 1)
        # self.batch_mask[:, :, :] = F.relu(self.transform_basic(Image.open(path_masks_list[idx_slice]), 1)
        #                                              - self.label_threshold).sign()

    def processing_training(self, path_images, path_masks):
        '''
        :param path_images_list: list of image path
        :param path_masks_list: list of mask path
        :return:
        '''
        # C D H W
        __image_temp = Image.open(path_images)
        __mask_temp = Image.open(path_masks)
        # print(path_images_list)
        ### coefficients
        h, w = __image_temp.size
        coef_RandomHorizontalFlip = random.random()
        coef_affine_angle = random.uniform(-10.0, 10.0)
        coef_affine_shear = random.uniform(-10.0, 10.0)
        ###

        # batch_image_temp = torch.zeros((1, self.num_slices_3d, h, w))
        # batch_mask_temp = torch.zeros((3, self.num_slices_3d, h, w))

        if coef_RandomHorizontalFlip > 0.5:
            __image_temp = transforms.functional.hflip(__image_temp)
            __mask_temp = transforms.functional.hflip(__mask_temp)

        __image_temp = transforms.functional.affine(__image_temp, coef_affine_angle, [0, 0], 1.0, coef_affine_shear,
                                         resample=PIL.Image.NEAREST)
        __mask_temp = transforms.functional.affine(__mask_temp, coef_affine_angle, [0, 0], 1.0, coef_affine_shear,
                                         resample=PIL.Image.NEAREST)

        __top, __bottom, __left, __right = self.get_cropping_boundary(__mask_temp)

        #     ##########
        #     batch_mask_np[idx_slice, :, :] = np.copy(np.array(__mask_temp_list[idx_slice])[:,:,1])
        #     ##########
        # batch_mask_np[batch_mask_np <= self.label_threshold] = 0
        # np.sum(batch_mask_np, axis=0, keepdims=False)
        # merged_label = np.nonzero(batch_mask_np)
        # __top2 = merged_label[0].min()
        # __bottom2 = merged_label[0].max()
        # __left2 = merged_label[1].min()
        # __right2 = merged_label[1].max()

        __top_limit = int(__top * 0.8)
        __bottom_limit = __bottom + int((h-__bottom)*0.2)
        __left_limit = int(__left*0.7)
        __right_limit = __right + int((w-__right)*0.3)

        if_height_was_found = False
        while not (if_height_was_found):

            __top_to_crop = round(__top_limit * random.random())
            __left_to_crop = round(__left_limit * random.random())

            __cropping_size_min = max([(__bottom_limit - __top_to_crop), (__right_limit - __left_to_crop)])
            __cropping_size_max = min([(h - __top_to_crop), (w - __left_to_crop)])
            if (__cropping_size_max >= __cropping_size_min):
                __cropping_size_coeff = random.random()
                __cropping_size = (__cropping_size_coeff * __cropping_size_min) + (
                            (1 - __cropping_size_coeff) * __cropping_size_max)
                if_height_was_found = True

        __image_temp = transforms.functional.crop(__image_temp, __top_to_crop, __left_to_crop, __cropping_size, __cropping_size)
        # __image_temp = transforms.functional.center_crop(__image_temp, __cropping_size)

        # resize, Dafault filter is PIL.Image.BILINEAR
        __image_temp = transforms.functional.resize(__image_temp, (self.image_size, self.image_size))

        __image_temp = self.colorJitter(__image_temp)

        __image_temp = self.tt(__image_temp)

        __image_temp.unsqueeze(1)

        self.batch_image = __image_temp

    def processing_data_id(self, idx):
        '''
        For 3d data
        :param idx:
        :return:
        '''
        # 2d data
        return os.path.split(self.csv_file.iloc[idx, 2])[-1]
        # return self.csv_file.iloc[idx, 1]


    def __getitem__(self, idx):
        # print("__getitem__!!!")
        if torch.is_tensor(idx):
            idx = idx.tolist()

        self.transform(self.csv_file.iloc[idx, 2], self.csv_file.iloc[idx, 3])

        # size: [1, 256, 256], [3,  256, 256], [7], [7]
        # sample = {'image': batch_image, 'mask': batch_mask, 'class': self.csv_file.iloc[idx, 0]}
        # mask[0]: background, mask[1]: rectum + rectal cancer, mask[2]: rectal cancer
        # image, classification label, patient number (2D)
        # NCHW
        return self.batch_image, float(self.csv_file.iloc[idx, 0]), self.processing_data_id(idx)



class CustomDataset_without_mask_2D(Dataset):

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

        # print("BEFORE2: " + str(image.size()))

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
            # print("batch_image-INIT: "+str(batch_image.size()))
            # image with smaller slice #(idx_slice) -> closer to head
            for idx_slice in range(1, len(dir_image)):
                batch_image = torch.cat([batch_image, self.transform_basic(Image.open(dir_image[idx_slice]))], dim=1)
            # print("after for loop: " + str(batch_image.size()))
        elif self.transform is True:
            # print(len(dir_image))
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


class CustomDataset_2D_with_mask(Dataset):

    def __init__(self, csv_file, transform=False, image_size = 256,  mask_ratio = 4.0,
                label_threshold=0.5, channel_for_cropping = 1, data_type = np.float32):

        self.csv_file = pd.read_csv(csv_file)
        self.image_size = image_size
        self.data_type = data_type
        self.channel_for_cropping = channel_for_cropping
        # ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)
        # self.colorJitter = transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)
        self.colorJitter = transforms.ColorJitter(0.2, 0.2, 0.2, 0.2)
        self.tt = transforms.ToTensor()
        self.label_threshold = label_threshold
        self.mask_ratio = mask_ratio

        if transform:
            self.transform = self.processing_training
        else:
            self.transform = self.processing_validation

    def __len__(self):
        return len(self.csv_file)

    def get_cropping_boundary(self, mask):
        '''
        self.cropping_boundary: [top, bottom, left, right]
        :param mask: PIL,  [H, W, C]
        '''
        __mask_np_temp = np.copy(np.array(mask)[:, :, 1])
        # __mask_np_temp[__mask_np_temp > self.label_threshold] = 1
        __mask_np_temp[__mask_np_temp <= self.label_threshold] = 0
        __mask_np_temp = np.nonzero(__mask_np_temp)

        return [__mask_np_temp[0].min(), __mask_np_temp[0].max(), __mask_np_temp[1].min(), __mask_np_temp[1].max()]

    def transform_basic(self, image, image_ratio):
        '''

        :param image: PIL
        :param resize_ratio: int or float
        :return:
        '''

        # resize, Dafault filter is PIL.Image.BILINEAR
        image = transforms.functional.resize(image, (int(self.image_size/image_ratio),
                                                   int(self.image_size/image_ratio)))
        # To Tensor
        image = self.tt(image)

        return image

    def processing_validation(self, path_images, path_masks):
    # def processing_validation(self, path_images_list, path_masks_list):
        '''
        :param path_images: sorted
        :param path_masks: sorted
        :return:
        '''

        # C H W
        self.batch_image = self.transform_basic(Image.open(path_images), 1)
        self.batch_mask = self.transform_basic(Image.open(path_masks), self.mask_ratio)
        # self.batch_mask[:, :, :] = F.relu(self.transform_basic(Image.open(path_masks_list[idx_slice]), 1)
        #                                              - self.label_threshold).sign()

    def processing_training(self, path_images, path_masks):
        '''
        :param path_images_list: list of image path
        :param path_masks_list: list of mask path
        :return:
        '''
        # C D H W
        __image_temp = Image.open(path_images)
        __mask_temp = Image.open(path_masks)
        # print(path_images_list)
        ### coefficients
        h, w = __image_temp.size
        coef_RandomHorizontalFlip = random.random()
        coef_affine_angle = random.uniform(-10.0, 10.0)
        coef_affine_shear = random.uniform(-10.0, 10.0)
        ###

        # batch_image_temp = torch.zeros((1, self.num_slices_3d, h, w))
        # batch_mask_temp = torch.zeros((3, self.num_slices_3d, h, w))

        if coef_RandomHorizontalFlip > 0.5:
            __image_temp = transforms.functional.hflip(__image_temp)
            __mask_temp = transforms.functional.hflip(__mask_temp)

        __image_temp = transforms.functional.affine(__image_temp, coef_affine_angle, [0, 0], 1.0, coef_affine_shear,
                                         resample=PIL.Image.NEAREST)
        __mask_temp = transforms.functional.affine(__mask_temp, coef_affine_angle, [0, 0], 1.0, coef_affine_shear,
                                         resample=PIL.Image.NEAREST)

        __top, __bottom, __left, __right = self.get_cropping_boundary(__mask_temp)

        #     ##########
        #     batch_mask_np[idx_slice, :, :] = np.copy(np.array(__mask_temp_list[idx_slice])[:,:,1])
        #     ##########
        # batch_mask_np[batch_mask_np <= self.label_threshold] = 0
        # np.sum(batch_mask_np, axis=0, keepdims=False)
        # merged_label = np.nonzero(batch_mask_np)
        # __top2 = merged_label[0].min()
        # __bottom2 = merged_label[0].max()
        # __left2 = merged_label[1].min()
        # __right2 = merged_label[1].max()

        __top_limit = int(__top * 0.15)
        __bottom_limit = __bottom + int((h-__bottom)*0.85)
        __left_limit = int(__left*0.15)
        __right_limit = __right + int((w-__right)*0.85)
        # __top = min(__top)
        # __bottom = max(__bottom)
        # __left = min(__left)
        # __right = max(__right)
        __center = int(h/2.0)
        __cropping_size_min = max( [__center-__top_limit, __bottom_limit-__center, __center-__left_limit, __right_limit-__center] )*2
        __cropping_size_coeff = random.random()
        __cropping_size = (__cropping_size_coeff*__cropping_size_min)+((1-__cropping_size_coeff)*h)

        # __image_temp_list[idx_slice] = transforms.functional.crop(__image_temp_list[idx_slice],
        #                                             __top_to_crop, __left_to_crop, __cropping_size, __cropping_size)
        # __image_temp_list[idx_slice] = transforms.functional.center_crop(__image_temp_list[idx_slice], (__cropping_size*2))
        __image_temp = transforms.functional.center_crop(__image_temp, __cropping_size)
        __mask_temp = transforms.functional.center_crop(__mask_temp, __cropping_size)

        # resize, Dafault filter is PIL.Image.BILINEAR
        __image_temp = transforms.functional.resize(__image_temp, (self.image_size, self.image_size))
        __mask_temp = transforms.functional.resize(__mask_temp, (int(self.image_size/self.mask_ratio), int(self.image_size/self.mask_ratio)))

        __image_temp = self.colorJitter(__image_temp)
        __mask_temp = self.colorJitter(__mask_temp)

        __image_temp = self.tt(__image_temp)
        __mask_temp = self.tt(__mask_temp)

        # __image_temp.unsqueeze(1)

        self.batch_image = __image_temp
        self.batch_mask = __mask_temp

    def processing_data_id(self, idx):
        '''
        For 3d data
        :param idx:
        :return:
        '''
        # 2d data
        return os.path.split(self.csv_file.iloc[idx, 2])[-1]
        # return self.csv_file.iloc[idx, 1]


    def __getitem__(self, idx):
        # print("__getitem__!!!")
        if torch.is_tensor(idx):
            idx = idx.tolist()

        self.transform(self.csv_file.iloc[idx, 2], self.csv_file.iloc[idx, 3])
        self.batch_mask[self.batch_mask < 0.5] = 0
        self.batch_mask[self.batch_mask >= 0.5] = 1
        # size: [1, 256, 256], [3,  256, 256], [7], [7]
        # sample = {'image': batch_image, 'mask': batch_mask, 'class': self.csv_file.iloc[idx, 0]}
        # mask[0]: background, mask[1]: rectum + rectal cancer, mask[2]: rectal cancer
        # image, classification label, patient number (2D)
        # NCHW
        return self.batch_image, self.batch_mask, float(self.csv_file.iloc[idx, 0]), self.processing_data_id(idx)




# # #
# # #
# # #
# # 3d
# # csv_tr = 'D:/Rectum_exp/Data/data_path_3d_new/3D_training_fold_1_ex.csv'
# # csv_val = 'D:/Rectum_exp/Data/data_path_3d_new/3D_validation_fold_1_ex.csv'
#
# # 2d
# csv_tr = 'D:/Rectum_exp/Data/data_path_3d_new/2D_training_fold_1_ex.csv'
# csv_val = 'D:/Rectum_exp/Data/data_path_3d_new/2D_validation_fold_1_ex.csv'
#
# # ds_tr = CustomDataset(csv_tr, transform=True)
# # ds_val = CustomDataset(csv_val, transform=False)
# ds_tr = CustomDataset_2D_no_mask_center_crop(csv_tr, transform=False)
# # ds_val = CustomDataset_3D_no_mask(csv_val, transform=False)
# for i in range(len(ds_tr)):
#     image, cls, patient_number = ds_tr[i]
#
#     print(i)
#     print(image.size())
#     print(cls)
#     print(patient_number)
#     # print(sample[0])
#     # print(sample)
#     # print(sample['image'].size(), sample['mask'].size(), sample['class'])
#     # print(sample[0].size(), sample[1].size(), sample[2], sample[3])
#     # print(sample[0].size(), sample[1], sample[2])
#     fig = plt.figure()
#     # for i in range(6):
#     #     fig.add_subplot(2, 3, (i+1))
#     #     plt.imshow(image[0, i, :, :], cmap='Greys_r')
#     plt.imshow(image[0,:,:], cmap='Greys_r')
#     # fig.add_subplot(1, 3, 2)
#     # plt.imshow(sample[1][1, 3, :, :], cmap='Greys_r')
#     # fig.add_subplot(1, 3, 3)
#     # plt.imshow(sample[1][2, 3, :, :], cmap='Greys_r')
#     # plt.title(str(int(sample[2]))+" / "+str(sample[3]))
#     plt.show()
