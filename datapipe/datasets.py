import random
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import Dataset

import rasterio
import csv
import os

def create_dataset(dataset_config):
    if dataset_config['type'] == 'cloudremoval': # For cloud removal task
        dataset = CloudRemovalDataset(**dataset_config['params'])
    else:
        raise NotImplementedError(dataset_config['type'])

    return dataset

def get_train_val_test_filelists(listpath):

    csv_file = open(listpath, "r")
    list_reader = csv.reader(csv_file)

    train_filelist = []
    val_filelist = []
    test_filelist = []
    for f in list_reader:
        line_entries = f
        if line_entries[0] == '1':
            train_filelist.append(line_entries)
        elif line_entries[0] == '2':
            val_filelist.append(line_entries)
        elif line_entries[0] == '3':
            test_filelist.append(line_entries)

    csv_file.close()

    return train_filelist, val_filelist, test_filelist

class CloudRemovalDataset(Dataset):
    def __init__(
            self,
            dir_path,
            mode,
            length=None,
            need_path=False,
            ):
        super().__init__()

        self.need_path = need_path
        
        self.iter_i = 0
        self.dir_path = dir_path
    
        self.mode = mode
        self.file_paths = []
        self.image_name = []
        self.clip_min = [[-25.0, -32.5], [0 for _ in range(13)], [0 for _ in range(13)]]
        self.clip_max = [[0, 0], [10000 for _ in range(13)], [10000 for _ in range(13)]]

        self.max_val = 1
        self.scale = 10000

        train_filelist, val_filelist, test_filelist = get_train_val_test_filelists(os.path.join(self.dir_path,'splits.csv'))

        if self.mode == 'train':
            self.tile_list = train_filelist
        elif self.mode == 'val':
            self.tile_list = val_filelist
        elif self.mode == 'test':
            self.tile_list = test_filelist
        self.file_paths_all = self.tile_list
        self.length = len(self.file_paths_all)

        image_roi_list = [tile[1].split('/')[0] for tile in self.tile_list]
        image_chunk_list = [tile[1].split('/')[1] for tile in self.tile_list]
        image_name_list = [tile[-1] for tile in self.tile_list]

        for idx in range(len(image_name_list)):
            image_s1_path = os.path.join(self.dir_path, image_roi_list[idx], image_chunk_list[idx], image_name_list[idx])
            image_s2_path = image_s1_path.replace("_s1", "_s2")
            image_s2_path = image_s2_path.replace("s1_", "s2_")
            image_s2cloudy_path = image_s1_path.replace("_s1", "_s2_cloudy")
            image_s2cloudy_path = image_s2cloudy_path.replace("s1_", "s2_cloudy_")

            self.file_paths.append(
                [image_s1_path, image_s2_path, image_s2cloudy_path])
            self.image_name.append(image_name_list[idx])

        self.augment_rotation_param = np.random.randint(
            0, 4, len(self.file_paths))
        self.augment_flip_param = np.random.randint(0, 3, len(self.file_paths))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        s1_path, s2_path, s2cloudy_path = self.file_paths[index][0], self.file_paths[index][1], self.file_paths[index][2]

        s1_data = self.get_sar_image(s1_path).astype('float32')
        s2_data = self.get_opt_image(s2_path).astype('float32')
        s2cloudy_data = self.get_opt_image(s2cloudy_path).astype('float32')
        
        s1_data = self.get_normalized_data(s1_data, data_type=1,index=index)
        s2_data = self.get_normalized_data(s2_data, data_type=2,index=index)
        s2cloudy_data = self.get_normalized_data(s2cloudy_data, data_type=3,index=index)

        im_path = self.file_paths[index]
        out_dict = {'gt':s2_data, }
        out_dict['cloudy'] = s2cloudy_data
        out_dict['s1'] = s1_data

        if self.need_path:
            out_dict['path'] = im_path

        return out_dict

    def reset_dataset(self):
        self.file_paths = random.sample(self.file_paths_all, self.length)

    def get_opt_image(self, path):

        src = rasterio.open(path, 'r', driver='GTiff')
        image = src.read()
        src.close()
        image[np.isnan(image)] = np.nanmean(image)  # fill holes and artifacts

        return image

    def get_sar_image(self, path):

        src = rasterio.open(path, 'r', driver='GTiff')
        image = src.read()
        src.close()
        image[np.isnan(image)] = np.nanmean(image)  # fill holes and artifacts

        return image

    def get_normalized_data(self, data_image, data_type, index):
        if self.mode == 'train':
            if not self.augment_flip_param[index] == 0:
                data_image = np.flip(data_image, self.augment_flip_param[index])
            if not self.augment_rotation_param[index] == 0:
                data_image = np.rot90(
                    data_image, self.augment_rotation_param[index], (1, 2)) 

        # SAR
        if data_type == 1:
            for channel in range(len(data_image)):
                data_image[channel] = np.clip(data_image[channel], self.clip_min[data_type - 1][channel], self.clip_max[data_type - 1][channel])
                data_image[channel] -= self.clip_min[data_type - 1][channel]
                data_image[channel] = self.max_val * (data_image[channel] / (self.clip_max[data_type - 1][channel] - self.clip_min[data_type - 1][channel]))
           
            data_image = torch.from_numpy((data_image.copy())).float()

            mean = torch.as_tensor([0.5, 0.5],
                            dtype=data_image.dtype, device=data_image.device)
            std = torch.as_tensor([0.5, 0.5],
                            dtype=data_image.dtype, device=data_image.device)
            if mean.ndim == 1:
                mean = mean.view(-1, 1, 1)
            if std.ndim == 1:
                std = std.view(-1, 1, 1)
            data_image.sub_(mean).div_(std)
        
        # OPT
        elif data_type == 2 or data_type == 3:
            for channel in range(len(data_image)):
                data_image[channel] = np.clip(data_image[channel], self.clip_min[data_type - 1][channel], self.clip_max[data_type - 1][channel])
            data_image /= self.scale

            data_image = torch.from_numpy((data_image.copy())).float()

            mean = torch.as_tensor([0.5 for _ in range(len(data_image))],
                            dtype=data_image.dtype, device=data_image.device)
            std = torch.as_tensor([0.5 for _ in range(len(data_image))],
                            dtype=data_image.dtype, device=data_image.device)
            if mean.ndim == 1:
                mean = mean.view(-1, 1, 1)
            if std.ndim == 1:
                std = std.view(-1, 1, 1)
            data_image.sub_(mean).div_(std)

        return data_image