from glob import glob
import torch

import os
import yaml
import numpy as np

def mask_op(data, x_min, x_max):
    mask = (data > x_min) & (data < x_max)
    return mask

def get_mask(pc, lims):
    mask_x = mask_op(pc[:, 0], lims[0][0] + 0.0001, lims[0][1] - 0.0001)
    mask_y = mask_op(pc[:, 1], lims[1][0] + 0.0001, lims[1][1] - 0.0001)
    mask_z = mask_op(pc[:, 2], lims[2][0] + 0.0001, lims[2][1] - 0.0001)
    mask = (mask_x) & (mask_y) & mask_z
    return mask


def unpack(compressed):
    ''' given a bit encoded voxel grid, make a normal voxel grid out of it.  '''
    uncompressed = np.zeros(compressed.shape[0] * 8, dtype=np.uint8)
    uncompressed[::8] = compressed[:] >> 7 & 1
    uncompressed[1::8] = compressed[:] >> 6 & 1
    uncompressed[2::8] = compressed[:] >> 5 & 1
    uncompressed[3::8] = compressed[:] >> 4 & 1
    uncompressed[4::8] = compressed[:] >> 3 & 1
    uncompressed[5::8] = compressed[:] >> 2 & 1
    uncompressed[6::8] = compressed[:] >> 1 & 1
    uncompressed[7::8] = compressed[:] & 1

    return uncompressed

class DSEC(torch.utils.data.Dataset):
    CLASSES = ('unlabeled',
               'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle',
               'person', 'bicyclist', 'motorcyclist', 'road',
               'parking', 'sidewalk', 'other-ground', 'building', 'fence',
               'vegetation', 'trunk', 'terrain', 'pole', 'traffic-sign')

    def __init__(self, data_root, data_config_file, setname,
                 lims,
                 sizes,
                 augmentation=False,
                 shuffle_index=False):
        self.data_root = data_root
        self.data_config = yaml.safe_load(open(data_config_file, 'r'))

        self.sequences = {
                'interlaken_00' : [51549876996000, 52174199996000],
                'interlaken_01' : [51549876996000, 52174199996000],
                'thun_00' : [51549876996000, 52174199996000],
                'thun_01' : [51549876996000, 52174199996000],
                'zurich_city_00' : [51549876996000, 52174199996000],
                'zurich_city_01' : [51549876996000, 52174199996000],
                'zurich_city_02' : [51549876996000, 52174199996000],
                'zurich_city_03' : [51549876996000, 52174199996000],
                'zurich_city_04' : [51549876996000, 52174199996000],
                'zurich_city_05' : [51549876996000, 52174199996000],
                'zurich_city_06' : [51549876996000, 52174199996000],
                'zurich_city_07' : [51549876996000, 52174199996000],
                'zurich_city_08' : [51549876996000, 52174199996000],
                'zurich_city_09' : [51549876996000, 52174199996000],
                'zurich_city_10' : [51549876996000, 52174199996000],
                'zurich_city_11' : [51549876996000, 52174199996000],
                'zurich_city_12' : [51549876996000, 52174199996000],
                'zurich_city_13' : [51549876996000, 52174199996000],
                'zurich_city_14' : [51549876996000, 52174199996000],
                'zurich_city_15' : [51549876996000, 52174199996000]
            }
        
        self.setname = setname
        self.labels = self.data_config['labels']
        self.learning_map = self.data_config["learning_map"]

        self.learning_map_inv = self.data_config["learning_map_inv"]
        self.color_map = self.data_config['color_map']

        self.lims = lims
        self.sizes = sizes
        self.augmentation = augmentation
        self.shuffle_index = shuffle_index

        self.filepaths = {}
        print(f"=> Parsing DSEC {self.setname}")
        self.get_filepaths()
        self.num_files_ = len(self.filepaths['occupancy'])
        print(f"Is aug: {self.augmentation}")

    def get_filepaths(self):
        # fill in with names, checking that all sequences are complete
        self.filepaths['occupancy'] = []

        for seq in self.sequences:
            print("parsing seq {}".format(seq))

            self.filepaths['occupancy'] += glob(os.path.join(self.data_root, 'data', seq, 'voxels', '*.bin'))
        
        
    def get_data(self, idx):
        data_collection = {}

        typ = 'occupancy'

        scan_data = unpack(np.fromfile(self.filepaths[typ][idx], dtype=np.uint8))
        print("================")
        print(self.filepaths[typ][idx])
        print(len(scan_data))

        scan_data = scan_data.reshape((self.sizes[0], self.sizes[1], self.sizes[2]))
        scan_data = scan_data.astype(np.float32)

        data_collection[typ] = torch.from_numpy(scan_data)


        points_path = self.filepaths['occupancy'][idx].replace('voxels', 'velodyne')
        points = np.fromfile(points_path, dtype=np.float32)
        points = points.reshape((-1, 4))

        if self.lims:
            filter_mask = get_mask(points, self.lims)
            points = points[filter_mask]

        data_collection['points'] = torch.from_numpy(points)

        return data_collection

    def __len__(self):
        return self.num_files_

    def get_n_classes(self):
        return len(self.learning_map_inv)

    def get_inv_remap_lut(self):
        '''
        remap_lut to remap classes of semantic kitti for training...
        :return:
        '''
        # make lookup table for mapping
        maxkey = max(self.learning_map_inv.keys())

        # +100 hack making lut bigger just in case there are unknown labels
        remap_lut = np.zeros((maxkey + 1), dtype=np.int32)
        remap_lut[list(self.learning_map_inv.keys())] = list(self.learning_map_inv.values())

        return remap_lut

    def to_color(self, label):
        # put label in original values
        label = DSEC.map(label, self.learning_map_inv)
        # put label in color
        return DSEC.map(label, self.color_map)

    def get_xentropy_class_string(self, idx):
        return self.labels[self.learning_map_inv[idx]]

    def __getitem__(self, idx):
        return self.get_data(idx), idx


