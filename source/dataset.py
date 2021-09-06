"""
Some parts of this code (preprocess.py) have been borrowed from from https://github.com/guiggh/hand_pose_action
"""
import os
import pickle

import torch
import yaml
import trimesh
import numpy as np
from PIL import Image

from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import cv2
import random
from vis_utils.vis_utils import *

from positional_encodings import PositionalEncoding1D, PositionalEncoding2D, PositionalEncoding3D


def _assert_exist(p):
    msg = 'File does not exists: %s' % p
    assert os.path.exists(p), msg


jointsMapManoToSimple = [0,
                         13, 14, 15, 16,
                         1, 2, 3, 17,
                         4, 5, 6, 18,
                         10, 11, 12, 19,
                         7, 8, 9, 20]

VISIBLE_PARAM = 0.04


class HO3D_v2_Dataset_pickle(Dataset):

    def __init__(self, mode='train', root='../../dataset/HO3D_V2', cfg='train', loadit=False):
        ###
        # initial setting
        # 640*480 image to 32*32*5 box
        # only control depth_discretization parameter(=5) in cfg.yaml
        ###
        self.coord_change_mat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)

        self.name = cfg
        self.root = root
        self.loadit = loadit
        self.mode = mode

        self.transform = transforms.ColorJitter(0.5, 0.5, 0.5)
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        # load meshes to memory
        # object_root = os.path.join(self.root, 'Object_models')
        # self.objects = self.load_objects(object_root)

        # self.camera_pose = np.array(
        #     [[0.999988496304, -0.00468848412856, 0.000982563360594, 25.7],
        #      [0.00469115935266, 0.999985218048, -0.00273845880292, 1.22],
        #      [-0.000969709653873, 0.00274303671904, 0.99999576807, 3.902],
        #      [0, 0, 0, 1]])

        self.camera_pose = np.array(
            [[1., 0., 0., 0.],
             [0., 1., 0., 0.],
             [0., 0., 1., 0.],
             [0, 0, 0, 1]])

        self.camera_intrinsics = np.array([[1395.749023, 0, 935.732544],
                                           [0, 1395.749268, 540.681030],
                                           [0, 0, 1]])

        self.depth_intrinsics = np.array([[475.065948, 0, 315.944855],
                                           [0,  475.065857, 245.287079],
                                           [0, 0, 1]])

        p_enc_3d = PositionalEncoding3D(21)
        z = torch.zeros((1, 13, 13, 5, 21))
        self.pos_encoder = p_enc_3d(z)

        self.prev_frame_idx = 0
        self.counting_idx = 0

        if self.name is 'train':
            self.GT_0 = np.load("./preproc_data/true_hand_pose.npy", mmap_mode='r')
            self.GT_1 = np.load("./preproc_data/hand_mask_stck.npy", mmap_mode='r')
            self.GT_2 = np.load("./preproc_data/true_object_pose_stck.npy", mmap_mode='r')
            self.GT_3 = np.load("./preproc_data/object_mask_stck.npy", mmap_mode='r')
            self.GT_4 = np.load("./preproc_data/param_vis_stck.npy", mmap_mode='r')
            self.GT_5 = np.load("./preproc_data/vis_stck.npy", mmap_mode='r')
            self.GT_6 = np.load("./preproc_data/extra_hand_pose_stck.npy", mmap_mode='r')
            self.GT_7 = np.load("./preproc_data/flag_set_stck.npy", mmap_mode='r')


        if not loadit:

            # subjects = [1, 2, 3, 4, 5, 6]
            # subject = "Subject_1"
            # subject = os.path.join(root, 'Object_6D_pose_annotation_v1', subject)
            # actions = os.listdir(subject)

            subject_path = os.path.join(root, mode)
            subjects = os.listdir(subject_path)

            dataset = dict()
            dataset['train'] = dict()
            dataset['test'] = dict()

            for subject in subjects:
                subject = str(subject)

                dataset['train'][subject] = list()
                dataset['test'][subject] = list()

                rgb_set = list(os.listdir(os.path.join(root, mode, subject, 'rgb')))
                frames = len(rgb_set)
                # random.shuffle(rgb_set)

                data_split = int(frames * 7 / 8) + 1

                for i in range(frames):
                    if i < data_split:
                        dataset['train'][subject].append(rgb_set[i])
                    else:
                        dataset['test'][subject].append(rgb_set[i])

            print(yaml.dump(dataset))

            modes = ['train', 'test']
            for i in range(2):
                self.samples = dict()
                self.name = modes[i]

                for subject in list(dataset[modes[i]]):
                    self.samples[subject] = dict()

                    idx = 0
                    for frame in dataset[modes[i]][subject]:
                        sample = {
                            'subject': subject,
                            'frame_idx': frame[:-4],
                        }
                        self.samples[subject][idx] = sample
                        idx += 1

            #self.clean_data()
                self.samples = self.samples.values()
                self.save_samples(modes[i])

        else:
            self.samples = self.load_samples(mode)

            ### test meta data has missing annotation, only acquire images in 'train' folder ###
            #self.mode = 'train'

            self.sample_len = len(self.samples)
            self.subject_order = np.arange(self.sample_len)

            self.samples_reorder = dict()
            idx = 0
            for subject in self.subject_order:
                for k, v in self.samples[subject].items():
                    self.samples_reorder[idx] = v
                    idx += 1

            self.samples = self.samples_reorder
            self.sample_len = len(self.samples)


    def load_samples(self, mode):
        with open('../cfg/HO3D_v2/{}.pkl'.format(self.name), 'rb') as f:
            samples = pickle.load(f)
            return samples

    def save_samples(self, mode):
        with open('../cfg/HO3D_v2/{}.pkl'.format(self.name), 'wb') as f:
            pickle.dump(list(self.samples), f, pickle.HIGHEST_PROTOCOL)

    def clean_data(self):
        print("Size beforing cleaning: {}".format(len(self.samples.keys())))

        for key in list(self.samples):
            try:
                self.__getitem__(key)
            except Exception as e:
                print(e)
                print("Index failed: {}".format(key))
                del self.samples[key]

        self.samples = self.samples.values()

        print("Size after cleaning: {}".format(len(self.samples)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.preprocess(idx)

    def get_image(self, sample, flag_set):
        img = self.fetch_image(sample)
        img = self.transform(img)
        img = np.asarray(img.resize((416, 416), Image.ANTIALIAS), dtype=np.float32)
        if img.shape[-1] != 3:
            img = img[:, :, :-1]

        if flag_set[0]:
            img = np.flip(img, axis=0)
        if flag_set[1]:
            img = np.flip(img, axis=1)

        img = img / 255.
        # cv2.imshow("img in dataset", img)
        # cv2.waitKey(0)
        img = np.squeeze(np.transpose(img, (2, 0, 1)))

        return img

    # def get_depth(self, sample):
    #
    #     img = self.fetch_depth(sample)
    #     img_np = np.array(img)
    #
    #     if self.mode == 'train':
    #         img = self.transform(img)
    #     img = np.asarray(img.resize((416, 416), Image.ANTIALIAS), dtype=np.float32)
    #     #img = np.flip(img, axis=1)
    #     img = img / 255.
    #     # cv2.imshow("img in dataset", img)
    #     # cv2.waitKey(1)
    #     img = np.transpose(img, (2, 0, 1))
    #     return img
    #
    # def read_image(self, sample):
    #     file_name = sample['frame_idx'] + '.png'
    #     img_path = os.path.join(self.root, self.mode, sample['subject'], 'rgb', file_name)
    #
    #     img = cv2.imread(img_path)
    #     return img
    #
    def fetch_image(self, sample):
        file_name = sample['frame_idx'] + '.png'
        img_path = os.path.join(self.root, 'train', sample['subject'], 'rgb', file_name)
        _assert_exist(img_path)

        img = Image.open(img_path)
        return img
    #
    # def fetch_depth(self, sample):
    #     file_name = sample['frame_idx'] + '.png'
    #     img_path = os.path.join(self.root, self.mode, sample['subject'], 'depth', file_name)
    #     img = Image.open(img_path)
    #     return img

    def read_data(self, sample):

        file_name = sample['frame_idx'] + '.pkl'
        meta_path = os.path.join(self.root, 'train', sample['subject'], 'meta', file_name)
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)

        file_name = sample['frame_idx'] + '.png'
        img_path = os.path.join(self.root, 'train', sample['subject'], 'rgb', file_name)
        _assert_exist(img_path)
        rgb = cv2.imread(img_path)

        img_path = os.path.join(self.root, 'train', sample['subject'], 'depth', file_name)
        _assert_exist(img_path)
        depth_scale = 0.00012498664727900177
        depth = cv2.imread(img_path)

        dpt = depth[:, :, 2] + depth[:, :, 1] * 256
        dpt = dpt * depth_scale

        return rgb, dpt, meta


    def preprocess(self, idx):
        """
        objTrans: A 3x1 vector representing object translation
        objRot: A 3x1 vector representing object rotation in axis-angle representation
        handPose: A 48x1 vector represeting the 3D rotation of the 16 hand joints including the root joint in axis-angle representation. The ordering of the joints follow the MANO model convention (see joint_order.png) and can be directly fed to MANO model.
        handTrans: A 3x1 vector representing the hand translation
        handBeta: A 10x1 vector representing the MANO hand shape parameters
        handJoints3D: A 21x3 matrix representing the 21 3D hand joint locations
        objCorners3D: A 8x3 matrix representing the 3D bounding box corners of the object
        objCorners3DRest: A 8x3 matrix representing the 3D bounding box corners of the object before applying the transformation
        objName: Name of the object as given in YCB dataset
        objLabel: Object label as given in YCB dataset
        camMat: Intrinsic camera parameters
        """
        #idx = idx % (self.sample_len)
        sample = self.samples[idx]

        true_hand_pose = torch.from_numpy(self.GT_0[idx, :])
        hand_mask = torch.from_numpy(self.GT_1[idx, :])
        true_object_pose = torch.from_numpy(self.GT_2[idx, :])
        object_mask = torch.from_numpy(self.GT_3[idx, :])
        param_vis = torch.from_numpy(self.GT_4[idx, :])
        vis = torch.from_numpy(self.GT_5[idx, :])
        extra_hand_pose = torch.from_numpy(self.GT_6[idx, :])
        flag_set = torch.from_numpy(self.GT_7[idx, :])

        image = None
        image = torch.from_numpy(self.get_image(sample, flag_set))
        if int(image.shape[0]) != 3:
            print("image shape wrong")
            image = image[:-1, :, :]
        image = self.normalize(image)

        return image, true_hand_pose, hand_mask, true_object_pose, object_mask, param_vis, vis, extra_hand_pose



    def load_objects(self, obj_root):
        object_names = ['juice', 'liquid_soap', 'milk', 'salt']
        all_models = {}
        for obj_name in object_names:
            obj_path = os.path.join(obj_root, '{}_model'.format(obj_name),
                                    '{}_model.ply'.format(obj_name))
            mesh = trimesh.load(obj_path)
            corners = trimesh.bounds.corners(mesh.bounding_box.bounds)
            all_models[obj_name] = {
                'corners': corners
            }
        return all_models

    def get_skeleton(self, sample, skel_root):
        skeleton_path = os.path.join(skel_root, sample['subject'],
                                     sample['action_name'], sample['seq_idx'],
                                     'skeleton.txt')
        #print('Loading skeleton from {}'.format(skeleton_path))
        skeleton_vals = np.loadtxt(skeleton_path)
        skeleton = skeleton_vals[:, 1:].reshape(skeleton_vals.shape[0], 21,
                                                -1)[sample['frame_idx']]
        return skeleton

    def get_object_pose(self, sample, obj_root):
        seq_path = os.path.join(obj_root, sample['subject'], sample['action_name'],
                                sample['seq_idx'], 'object_pose.txt')
        with open(seq_path, 'r') as seq_f:
            raw_lines = seq_f.readlines()
        raw_line = raw_lines[sample['frame_idx']]
        line = raw_line.strip().split(' ')
        trans_matrix = np.array(line[1:]).astype(np.float32)
        trans_matrix = trans_matrix.reshape(4, 4).transpose()
        # print('Loading obj transform from {}'.format(seq_path))
        return trans_matrix

    def downsample_points(self, points, depth):

        downsample_ratio_x = 640 / 416.
        downsample_ratio_y = 480 / 416.

        x = points[0] / downsample_ratio_x
        y = points[1] / downsample_ratio_y
        z = depth / 10.  # converting to centimeters

        downsampled_x = x / 32
        downsampled_y = y / 32
        downsampled_z = z / 15

        return downsampled_x, downsampled_y, downsampled_z

    def upsample_points(self, points, depth):

        downsample_ratio_x = 640 / 416.
        downsample_ratio_y = 480 / 416.

        u = points[0] * downsample_ratio_x
        v = points[1] * downsample_ratio_y
        z = depth * 10.  # converting to millimeters

        return u, v, z

    def get_cell(self, root, depth):

        downsampled_x, downsampled_y, downsampled_z = self.downsample_points(root, depth)

        u = int(downsampled_x)
        v = int(downsampled_y)
        z = int(downsampled_z)

        return (u, v, z)

    def compute_offset(self, points, cell):

        points_u, points_v, points_z = points
        points_u, points_v, points_z = self.downsample_points((points_u, points_v), points_z)
        cell_u, cell_v, cell_z = cell
        del_u = points_u - cell_u
        del_v = points_v - cell_v
        del_z = points_z - cell_z

        return del_u, del_v, del_z

    def get_box_3d_control_points(self, corners):

        # lines (0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4), (0,4), (1,5), (2,6), (3,7)

        edge_01 = (corners[0] + corners[1]) / 2.
        edge_12 = (corners[1] + corners[2]) / 2.
        edge_23 = (corners[2] + corners[3]) / 2.
        edge_30 = (corners[3] + corners[0]) / 2.
        edge_45 = (corners[4] + corners[5]) / 2.
        edge_56 = (corners[5] + corners[6]) / 2.
        edge_67 = (corners[6] + corners[7]) / 2.
        edge_74 = (corners[7] + corners[4]) / 2.
        edge_04 = (corners[0] + corners[4]) / 2.
        edge_15 = (corners[1] + corners[5]) / 2.
        edge_26 = (corners[2] + corners[6]) / 2.
        edge_37 = (corners[3] + corners[7]) / 2.

        center = np.mean(corners, axis=0)

        control_points = np.vstack((center, corners,
                                    edge_01, edge_12, edge_23, edge_30,
                                    edge_45, edge_56, edge_67, edge_74,
                                    edge_04, edge_15, edge_26, edge_37))

        return control_points

    def control_to_target(self, projected_points, points):
        # 2D location : projected_points[0, :]
        # depth : points[0, 2]
        root = projected_points[0, :]

        # get location of cell in (13, 13, 5)
        cell = self.get_cell(root, points[0, 2])
        u, v, z = cell
        if u > 12:
            u = 12
        if v > 12:
            v = 12
        if z > 4:
            z = 4
        cell = [u, v, z]

        points = projected_points[:, 0], projected_points[:, 1], points[:, 2]  # px, px, mm

        del_u, del_v, del_z = self.compute_offset(points, cell)

        return del_u, del_v, del_z, cell

    def target_to_control(self, del_u, del_v, del_z, cell):

        u, v, z = cell

        w_u = del_u + u
        w_v = del_v + v
        w_z = del_z + z

        w_u, w_v, w_z = self.upsample_points((w_u, w_v), w_z)
        points = np.vstack((w_u * 32, w_v * 32, np.ones_like(w_u)))
        y_hat = w_z * 15 * np.linalg.inv(self.camera_intrinsics).dot(points)

        return y_hat.T, points


class HO3D_v3_Dataset(Dataset):

    def __init__(self, mode='train', root='../../dataset/HO3D_v3/HO3D_v3', cfg='train', loadit=False):
        ###
        # initial setting
        # 640*480 image to 32*32*5 box
        # only control depth_discretization parameter(=5) in cfg.yaml
        ###
        self.coord_change_mat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)

        self.name = cfg
        self.root = root
        self.loadit = loadit
        self.mode = mode

        self.transform = transforms.ColorJitter(0.5, 0.5, 0.5)
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        # load meshes to memory
        # object_root = os.path.join(self.root, 'Object_models')
        # self.objects = self.load_objects(object_root)

        # self.camera_pose = np.array(
        #     [[0.999988496304, -0.00468848412856, 0.000982563360594, 25.7],
        #      [0.00469115935266, 0.999985218048, -0.00273845880292, 1.22],
        #      [-0.000969709653873, 0.00274303671904, 0.99999576807, 3.902],
        #      [0, 0, 0, 1]])

        self.camera_pose = np.array(
            [[1., 0., 0., 0.],
             [0., 1., 0., 0.],
             [0., 0., 1., 0.],
             [0, 0, 0, 1]])

        self.camera_intrinsics = np.array([[1395.749023, 0, 935.732544],
                                           [0, 1395.749268, 540.681030],
                                           [0, 0, 1]])

        self.depth_intrinsics = np.array([[475.065948, 0, 315.944855],
                                           [0,  475.065857, 245.287079],
                                           [0, 0, 1]])

        p_enc_3d = PositionalEncoding3D(21)
        z = torch.zeros((1, 13, 13, 5, 21))
        self.pos_encoder = p_enc_3d(z)

        self.prev_frame_idx = 0
        self.counting_idx = 0

        if not loadit:

            # subjects = [1, 2, 3, 4, 5, 6]
            # subject = "Subject_1"
            # subject = os.path.join(root, 'Object_6D_pose_annotation_v1', subject)
            # actions = os.listdir(subject)

            subject_path = os.path.join(root, mode)
            subjects = os.listdir(subject_path)

            dataset = dict()
            dataset['train'] = dict()
            dataset['test'] = dict()

            for subject in subjects:
                subject = str(subject)

                dataset['train'][subject] = list()
                dataset['test'][subject] = list()

                rgb_set = list(os.listdir(os.path.join(root, mode, subject, 'rgb')))
                frames = len(rgb_set)
                # random.shuffle(rgb_set)

                data_split = int(frames * 7 / 8) + 1

                for i in range(frames):
                    if i < data_split:
                        dataset['train'][subject].append(rgb_set[i])
                    else:
                        dataset['test'][subject].append(rgb_set[i])

            print(yaml.dump(dataset))

            modes = ['train', 'test']
            for i in range(2):
                self.samples = dict()
                self.name = modes[i]

                for subject in list(dataset[modes[i]]):
                    self.samples[subject] = dict()

                    idx = 0
                    for frame in dataset[modes[i]][subject]:
                        sample = {
                            'subject': subject,
                            'frame_idx': frame[:-4],
                        }
                        self.samples[subject][idx] = sample
                        idx += 1

            #self.clean_data()
                self.samples = self.samples.values()
                self.save_samples(modes[i])

        else:
            self.samples = self.load_samples(mode)

            ### test meta data has missing annotation, only acquire images in 'train' folder ###
            #self.mode = 'train'

            self.sample_len = len(self.samples)
            self.subject_order = np.arange(self.sample_len)

            self.samples_reorder = dict()
            idx = 0
            for subject in self.subject_order:
                for k, v in self.samples[subject].items():
                    self.samples_reorder[idx] = v
                    idx += 1

            self.samples = self.samples_reorder
            self.sample_len = len(self.samples)


    def load_samples(self, mode):
        with open('../cfg/HO3D_v3/{}.pkl'.format(self.name), 'rb') as f:
            samples = pickle.load(f)
            return samples

    def save_samples(self, mode):
        with open('../cfg/HO3D_v3/{}.pkl'.format(self.name), 'wb') as f:
            pickle.dump(list(self.samples), f, pickle.HIGHEST_PROTOCOL)

    def clean_data(self):
        print("Size beforing cleaning: {}".format(len(self.samples.keys())))

        for key in list(self.samples):
            try:
                self.__getitem__(key)
            except Exception as e:
                print(e)
                print("Index failed: {}".format(key))
                del self.samples[key]

        self.samples = self.samples.values()

        print("Size after cleaning: {}".format(len(self.samples)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.preprocess(idx)

    def get_image(self, sample, flag_set):
        img = self.fetch_image(sample)
        img = self.transform(img)
        img = np.asarray(img.resize((416, 416), Image.ANTIALIAS), dtype=np.float32)
        if img.shape[-1] != 3:
            img = img[:, :, :-1]

        if flag_set[0]:
            img = np.flip(img, axis=0)
        if flag_set[1]:
            img = np.flip(img, axis=1)

        img = img / 255.
        # cv2.imshow("img in dataset", img)
        # cv2.waitKey(0)
        img = np.squeeze(np.transpose(img, (2, 0, 1)))

        return img

    # def get_depth(self, sample):
    #
    #     img = self.fetch_depth(sample)
    #     img_np = np.array(img)
    #
    #     if self.mode == 'train':
    #         img = self.transform(img)
    #     img = np.asarray(img.resize((416, 416), Image.ANTIALIAS), dtype=np.float32)
    #     #img = np.flip(img, axis=1)
    #     img = img / 255.
    #     # cv2.imshow("img in dataset", img)
    #     # cv2.waitKey(1)
    #     img = np.transpose(img, (2, 0, 1))
    #     return img
    #
    # def read_image(self, sample):
    #     file_name = sample['frame_idx'] + '.png'
    #     img_path = os.path.join(self.root, self.mode, sample['subject'], 'rgb', file_name)
    #
    #     img = cv2.imread(img_path)
    #     return img
    #
    def fetch_image(self, sample):
        file_name = sample['frame_idx'] + '.jpg'
        img_path = os.path.join(self.root, 'train', sample['subject'], 'rgb', file_name)
        _assert_exist(img_path)

        img = Image.open(img_path)
        return img
    #
    # def fetch_depth(self, sample):
    #     file_name = sample['frame_idx'] + '.png'
    #     img_path = os.path.join(self.root, self.mode, sample['subject'], 'depth', file_name)
    #     img = Image.open(img_path)
    #     return img

    def read_data(self, sample):

        file_name = sample['frame_idx'] + '.pkl'
        meta_path = os.path.join(self.root, 'train', sample['subject'], 'meta', file_name)
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)

        file_name = sample['frame_idx'] + '.jpg'
        img_path = os.path.join(self.root, 'train', sample['subject'], 'rgb', file_name)
        _assert_exist(img_path)
        rgb = cv2.imread(img_path)

        file_name = sample['frame_idx'] + '.png'
        img_path = os.path.join(self.root, 'train', sample['subject'], 'depth', file_name)
        _assert_exist(img_path)
        depth_scale = 0.00012498664727900177
        depth = cv2.imread(img_path)

        dpt = depth[:, :, 2] + depth[:, :, 1] * 256
        dpt = dpt * depth_scale

        return rgb, dpt, meta


    def preprocess(self, idx):
        """
        objTrans: A 3x1 vector representing object translation
        objRot: A 3x1 vector representing object rotation in axis-angle representation
        handPose: A 48x1 vector represeting the 3D rotation of the 16 hand joints including the root joint in axis-angle representation. The ordering of the joints follow the MANO model convention (see joint_order.png) and can be directly fed to MANO model.
        handTrans: A 3x1 vector representing the hand translation
        handBeta: A 10x1 vector representing the MANO hand shape parameters
        handJoints3D: A 21x3 matrix representing the 21 3D hand joint locations
        objCorners3D: A 8x3 matrix representing the 3D bounding box corners of the object
        objCorners3DRest: A 8x3 matrix representing the 3D bounding box corners of the object before applying the transformation
        objName: Name of the object as given in YCB dataset
        objLabel: Object label as given in YCB dataset
        camMat: Intrinsic camera parameters
        """
        #idx = idx % (self.sample_len)
        sample = self.samples[idx]

        frame_idx = int(sample['frame_idx'])

        img, depth, meta = self.read_data(sample)
        # subject = sample['subject']

        objCorners = meta['objCorners3DRest']
        objCornersTrans = np.matmul(objCorners, cv2.Rodrigues(meta['objRot'])[0].T) + meta['objTrans']
        objCornersTrans = objCornersTrans.dot(self.coord_change_mat.T) * 1000.
        objcontrolPoints = self.get_box_3d_control_points(objCornersTrans)
        objKps = project_3D_points(meta['camMat'], objcontrolPoints, is_OpenGL_coords=False)

        handJoints3D = meta['handJoints3D']
        handJoints3D = handJoints3D.dot(self.coord_change_mat.T) * 1000.
        handKps = project_3D_points(meta['camMat'], handJoints3D, is_OpenGL_coords=False)

        handKps_ = handKps[jointsMapManoToSimple]
        handJoints3D_ = handJoints3D[jointsMapManoToSimple]
        handKps_ = np.round(handKps_).astype(np.int)
        visible = []
        for i in range(21):
            if handKps_[i][0] >= 640 or handKps_[i][1] >= 480:
                continue
            d_img = depth[handKps_[i][1], handKps_[i][0]]
            d_gt = handJoints3D_[i][-1]
            if np.abs(d_img - d_gt) < VISIBLE_PARAM:
                visible.append(i)
        """
        # depth_proc = np.copy(depth)
        # depth_proc[depth > 1.0] = 0.0
        # depthAnno = showHandJoints(depth_proc, handKps)
        imgAnno = showHandJoints_vis(img, handKps, vis=visible)
        # imgAnno = showHandJoints(img, handKps)
        # imgAnno = showObjJoints(imgAnno, objKps, lineThickness=2)
        rgb = img[:, :, [0, 1, 2]]
        cv2.imshow("rgb", rgb)
        anno = imgAnno[:, :, [0, 1, 2]]
        cv2.imshow("anno", anno)
        # cv2.imshow("depthAnno", depthAnno)
        cv2.waitKey(1)

        handJoints3D_vert = np.zeros((21, 3), dtype=np.float64)
        handJoints3D_vert[:, 1] = handJoints3D[:, 1] * -1
        handJoints3D_vert[:, 0] = handJoints3D[:, 0]
        handJoints3D_vert[:, -1] = handJoints3D[:, -1]
        handKps_vert = project_3D_points(meta['camMat'], handJoints3D_vert, is_OpenGL_coords=False)

        img = np.flip(img, axis=0)
        imgAnno = showHandJoints_vis(img, handKps_vert, vis=visible)
        # imgAnno = showHandJoints(img, handKps)
        # imgAnno = showObjJoints(imgAnno, objKps, lineThickness=2)
        rgb = img[:, :, [0, 1, 2]]
        cv2.imshow("rgb_vert", rgb)
        anno = imgAnno[:, :, [0, 1, 2]]
        cv2.imshow("anno_vert", anno)
        cv2.waitKey(0)
        """
        ### data augmentation ###

        flag_vert = np.random.randint(2)
        flag_hori = np.random.randint(2)
        flag_set = [flag_vert, flag_hori]

        handJoints3D_flip = np.copy(handJoints3D)
        objcontrolPoints_flip = np.copy(objcontrolPoints)
        if flag_vert:
            handJoints3D_flip[:, 1] = handJoints3D[:, 1] * -1
            objcontrolPoints_flip[:, 1] = objcontrolPoints[:, 1] * -1

        if flag_hori:
            handJoints3D_flip[:, 0] = handJoints3D[:, 0] * -1
            objcontrolPoints_flip[:, 0] = objcontrolPoints[:, 0] * -1

        objcontrolPoints = objcontrolPoints_flip
        handJoints3D = handJoints3D_flip

        handKps = project_3D_points(meta['camMat'], handJoints3D, is_OpenGL_coords=False)
        handKps = handKps[jointsMapManoToSimple]
        handJoints3D = handJoints3D[jointsMapManoToSimple]

        objKps = project_3D_points(meta['camMat'], objcontrolPoints, is_OpenGL_coords=False)

        ### object pose ###
        # get offset w.r.t top/left corner of the cell
        del_u, del_v, del_z, cell = self.control_to_target(objKps, objcontrolPoints)

        # object pose tensor
        true_object_pose = torch.zeros(21, 3, 5, 13, 13, dtype=torch.float32)
        u, v, z = cell
        pose = np.vstack((del_u, del_v, del_z)).T
        true_object_pose[:, :, z, v, u] = torch.from_numpy(pose)
        true_object_pose = true_object_pose.view(-1, 5, 13, 13)     # (63, 5, 13, 13)

        # object mask
        object_mask = torch.zeros(5, 13, 13, dtype=torch.float32)
        object_mask[z, v, u] = 1


        ### hand pose ###
        del_u, del_v, del_z, cell = self.control_to_target(handKps, handJoints3D)  # handKps : [215.8, 182.1] , ...   /  handJoints3D : [~, ~, 462.2] , ...
        # hand pose tensor
        true_hand_pose = torch.zeros(21, 3, 5, 13, 13, dtype=torch.float32)
        u, v, z = cell
        pose = np.vstack((del_u, del_v, del_z)).T
        true_hand_pose[:, :, z, v, u] = torch.from_numpy(pose)
        true_hand_pose = true_hand_pose.view(-1, 5, 13, 13)
        # hand mask
        hand_mask = torch.zeros(5, 13, 13, dtype=torch.float32)
        hand_mask[z, v, u] = 1

        # hand visibility per joint
        vis = np.zeros(21, dtype=np.float32)
        if len(visible) != 0:
            vis[np.array(visible)] = 1

        param_vis = torch.zeros(21, 5, 13, 13, dtype=torch.float32)
        param_vis[:, z, v, u] = torch.from_numpy(vis)
        param_vis = param_vis.view(-1, 5, 13, 13)

        image = None

        if self.loadit:
            image = torch.from_numpy(self.get_image(sample, flag_set))
            if int(image.shape[0]) != 3:
                print("image shape wrong")
                image = image[:-1, :, :]
            image = self.normalize(image)

        ### preprocess extrapolated keypoint GT ###
        if self.mode == 'train':
            prev_handKps = dict()
            prev_handJoints3D = dict()
            prev_vis = dict()

            if np.abs(frame_idx - self.prev_frame_idx) > 2:
                self.counting_idx = 0

            self.prev_frame_idx = frame_idx

            if self.counting_idx < 2:
                self.counting_idx += 1

                for i in range(2):
                    prev_handKps[i] = np.zeros((21, 2), dtype=np.float32)
                    prev_handJoints3D[i] = np.zeros((21, 3), dtype=np.float32)
                    prev_vis[i] = np.zeros(21, dtype=np.float32)
            else:
                for i in range(2):
                    prev_sample = self.samples[idx - (i + 1)]
                    img, depth, meta = self.read_data(prev_sample)

                    handJoints3D = meta['handJoints3D']
                    handJoints3D = handJoints3D.dot(self.coord_change_mat.T) * 1000.

                    ##################################################################

                    handKps_ = project_3D_points(meta['camMat'], handJoints3D, is_OpenGL_coords=False)

                    handKps_ = handKps_[jointsMapManoToSimple]
                    handJoints3D_ = handJoints3D[jointsMapManoToSimple]
                    handKps_ = np.round(handKps_).astype(np.int)
                    visible = []
                    for i in range(21):
                        if handKps_[i][0] >= 640 or handKps_[i][1] >= 480:
                            continue
                        d_img = depth[handKps_[i][1], handKps_[i][0]]
                        d_gt = handJoints3D_[i][-1]
                        if np.abs(d_img - d_gt) < VISIBLE_PARAM:
                            visible.append(i)
                    vis = np.zeros(21, dtype=np.float32)
                    if len(visible) != 0:
                        vis[np.array(visible)] = 1

                    ##################################################################
                    handJoints3D_flip = np.copy(handJoints3D)
                    if flag_vert:
                        handJoints3D_flip[:, 1] = handJoints3D[:, 1] * -1
                    if flag_hori:
                        handJoints3D_flip[:, 0] = handJoints3D[:, 0] * -1

                    handJoints3D = handJoints3D_flip

                    handKps = project_3D_points(meta['camMat'], handJoints3D, is_OpenGL_coords=False)

                    handKps = handKps[jointsMapManoToSimple]
                    handJoints3D = handJoints3D[jointsMapManoToSimple]

                    prev_handKps[i] = handKps
                    prev_handJoints3D[i] = handJoints3D
                    prev_vis[i] = vis[jointsMapManoToSimple]

            extra_handKps = 2 * prev_handKps[0] - prev_handKps[1]
            extra_handJoints3D = 2 * prev_handJoints3D[0] - prev_handJoints3D[1]

            mixed_handKps = np.copy(prev_handKps[0])
            mixed_handJoints3D = np.copy(prev_handJoints3D[0])

            visible_index = np.multiply(prev_vis[0], prev_vis[1]).astype(int)

            mixed_handKps[visible_index, :] = extra_handKps[visible_index, :]
            mixed_handJoints3D[visible_index, :] = extra_handJoints3D[visible_index, :]

            del_u, del_v, del_z, cell = self.control_to_target(mixed_handKps, mixed_handJoints3D)
            # hand pose tensor
            # index + del, with positional encoding
            del_u = torch.unsqueeze(torch.from_numpy(del_u), 0).type(torch.float32)
            del_v = torch.unsqueeze(torch.from_numpy(del_v), 0).type(torch.float32)
            del_z = torch.unsqueeze(torch.from_numpy(del_z), 0).type(torch.float32)

            enc_cell = self.pos_encoder[:, cell[0], cell[1], cell[2], :].type(torch.float32)

            extra_hand_pose = torch.cat((enc_cell, del_u, del_v, del_z), 0) # tensor (4, 21)

            return image, true_hand_pose, hand_mask, true_object_pose, object_mask, param_vis, vis, extra_hand_pose

        else:
            if np.abs(frame_idx - self.prev_frame_idx) > 2:
                flag_seq = True
            else:
                flag_seq = False

            self.prev_frame_idx = frame_idx

            return image, true_hand_pose, hand_mask, true_object_pose, object_mask, param_vis, vis, flag_seq


    def load_objects(self, obj_root):
        object_names = ['juice', 'liquid_soap', 'milk', 'salt']
        all_models = {}
        for obj_name in object_names:
            obj_path = os.path.join(obj_root, '{}_model'.format(obj_name),
                                    '{}_model.ply'.format(obj_name))
            mesh = trimesh.load(obj_path)
            corners = trimesh.bounds.corners(mesh.bounding_box.bounds)
            all_models[obj_name] = {
                'corners': corners
            }
        return all_models

    def get_skeleton(self, sample, skel_root):
        skeleton_path = os.path.join(skel_root, sample['subject'],
                                     sample['action_name'], sample['seq_idx'],
                                     'skeleton.txt')
        #print('Loading skeleton from {}'.format(skeleton_path))
        skeleton_vals = np.loadtxt(skeleton_path)
        skeleton = skeleton_vals[:, 1:].reshape(skeleton_vals.shape[0], 21,
                                                -1)[sample['frame_idx']]
        return skeleton

    def get_object_pose(self, sample, obj_root):
        seq_path = os.path.join(obj_root, sample['subject'], sample['action_name'],
                                sample['seq_idx'], 'object_pose.txt')
        with open(seq_path, 'r') as seq_f:
            raw_lines = seq_f.readlines()
        raw_line = raw_lines[sample['frame_idx']]
        line = raw_line.strip().split(' ')
        trans_matrix = np.array(line[1:]).astype(np.float32)
        trans_matrix = trans_matrix.reshape(4, 4).transpose()
        # print('Loading obj transform from {}'.format(seq_path))
        return trans_matrix

    def downsample_points(self, points, depth):

        downsample_ratio_x = 640 / 416.
        downsample_ratio_y = 480 / 416.

        x = points[0] / downsample_ratio_x
        y = points[1] / downsample_ratio_y
        z = depth / 10.  # converting to centimeters

        downsampled_x = x / 32
        downsampled_y = y / 32
        downsampled_z = z / 15

        return downsampled_x, downsampled_y, downsampled_z

    def upsample_points(self, points, depth):

        downsample_ratio_x = 640 / 416.
        downsample_ratio_y = 480 / 416.

        u = points[0] * downsample_ratio_x
        v = points[1] * downsample_ratio_y
        z = depth * 10.  # converting to millimeters

        return u, v, z

    def get_cell(self, root, depth):

        downsampled_x, downsampled_y, downsampled_z = self.downsample_points(root, depth)

        u = int(downsampled_x)
        v = int(downsampled_y)
        z = int(downsampled_z)

        return (u, v, z)

    def compute_offset(self, points, cell):

        points_u, points_v, points_z = points
        points_u, points_v, points_z = self.downsample_points((points_u, points_v), points_z)
        cell_u, cell_v, cell_z = cell
        del_u = points_u - cell_u
        del_v = points_v - cell_v
        del_z = points_z - cell_z

        return del_u, del_v, del_z

    def get_box_3d_control_points(self, corners):

        # lines (0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4), (0,4), (1,5), (2,6), (3,7)

        edge_01 = (corners[0] + corners[1]) / 2.
        edge_12 = (corners[1] + corners[2]) / 2.
        edge_23 = (corners[2] + corners[3]) / 2.
        edge_30 = (corners[3] + corners[0]) / 2.
        edge_45 = (corners[4] + corners[5]) / 2.
        edge_56 = (corners[5] + corners[6]) / 2.
        edge_67 = (corners[6] + corners[7]) / 2.
        edge_74 = (corners[7] + corners[4]) / 2.
        edge_04 = (corners[0] + corners[4]) / 2.
        edge_15 = (corners[1] + corners[5]) / 2.
        edge_26 = (corners[2] + corners[6]) / 2.
        edge_37 = (corners[3] + corners[7]) / 2.

        center = np.mean(corners, axis=0)

        control_points = np.vstack((center, corners,
                                    edge_01, edge_12, edge_23, edge_30,
                                    edge_45, edge_56, edge_67, edge_74,
                                    edge_04, edge_15, edge_26, edge_37))

        return control_points

    def control_to_target(self, projected_points, points):
        # 2D location : projected_points[0, :]
        # depth : points[0, 2]
        root = projected_points[0, :]

        # get location of cell in (13, 13, 5)
        cell = self.get_cell(root, points[0, 2])
        u, v, z = cell
        if u > 12:
            u = 12
        if v > 12:
            v = 12
        if z > 4:
            z = 4
        cell = [u, v, z]

        points = projected_points[:, 0], projected_points[:, 1], points[:, 2]  # px, px, mm

        del_u, del_v, del_z = self.compute_offset(points, cell)

        return del_u, del_v, del_z, cell

    def target_to_control(self, del_u, del_v, del_z, cell):

        u, v, z = cell

        w_u = del_u + u
        w_v = del_v + v
        w_z = del_z + z

        w_u, w_v, w_z = self.upsample_points((w_u, w_v), w_z)
        points = np.vstack((w_u * 32, w_v * 32, np.ones_like(w_u)))
        y_hat = w_z * 15 * np.linalg.inv(self.camera_intrinsics).dot(points)

        return y_hat.T, points


class HO3D_v2_Dataset_update(Dataset):

    def __init__(self, mode='train', root='../../dataset/HO3D_V2', cfg='train', loadit=False):
        ###
        # initial setting
        # 640*480 image to 32*32*5 box
        # only control depth_discretization parameter(=5) in cfg.yaml
        ###
        self.coord_change_mat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)

        self.name = cfg
        self.root = root
        self.loadit = loadit
        self.mode = mode

        self.transform = transforms.ColorJitter(0.5, 0.5, 0.5)
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        # load meshes to memory
        # object_root = os.path.join(self.root, 'Object_models')
        # self.objects = self.load_objects(object_root)

        # self.camera_pose = np.array(
        #     [[0.999988496304, -0.00468848412856, 0.000982563360594, 25.7],
        #      [0.00469115935266, 0.999985218048, -0.00273845880292, 1.22],
        #      [-0.000969709653873, 0.00274303671904, 0.99999576807, 3.902],
        #      [0, 0, 0, 1]])

        self.camera_pose = np.array(
            [[1., 0., 0., 0.],
             [0., 1., 0., 0.],
             [0., 0., 1., 0.],
             [0, 0, 0, 1]])

        self.camera_intrinsics = np.array([[1395.749023, 0, 935.732544],
                                           [0, 1395.749268, 540.681030],
                                           [0, 0, 1]])

        self.depth_intrinsics = np.array([[475.065948, 0, 315.944855],
                                           [0,  475.065857, 245.287079],
                                           [0, 0, 1]])

        p_enc_3d = PositionalEncoding3D(21)
        z = torch.zeros((1, 13, 13, 5, 21))
        self.pos_encoder = p_enc_3d(z)

        self.prev_frame_idx = 0
        self.counting_idx = 0

        if not loadit:

            # subjects = [1, 2, 3, 4, 5, 6]
            # subject = "Subject_1"
            # subject = os.path.join(root, 'Object_6D_pose_annotation_v1', subject)
            # actions = os.listdir(subject)

            subject_path = os.path.join(root, mode)
            subjects = os.listdir(subject_path)

            dataset = dict()
            dataset['train'] = dict()
            dataset['test'] = dict()

            for subject in subjects:
                subject = str(subject)

                dataset['train'][subject] = list()
                dataset['test'][subject] = list()

                rgb_set = list(os.listdir(os.path.join(root, mode, subject, 'rgb')))
                frames = len(rgb_set)
                # random.shuffle(rgb_set)

                # data_split = int(frames * 7 / 8) + 1
                # for i in range(frames):
                #     if i < data_split:
                #         dataset['train'][subject].append(rgb_set[i])
                #     else:
                #         dataset['test'][subject].append(rgb_set[i])

                data_split = int(frames * 1 / 4) + 1
                data_split_train = int(data_split * 7 / 8) + 1
                for i in range(frames):
                    if i < data_split_train:
                        dataset['train'][subject].append(rgb_set[i])
                    elif i < data_split:
                        dataset['test'][subject].append(rgb_set[i])

            print(yaml.dump(dataset))

            modes = ['train', 'test']
            for i in range(2):
                self.samples = dict()
                self.name = modes[i]

                for subject in list(dataset[modes[i]]):
                    self.samples[subject] = dict()

                    idx = 0
                    for frame in dataset[modes[i]][subject]:
                        sample = {
                            'subject': subject,
                            'frame_idx': frame[:-4],
                        }
                        self.samples[subject][idx] = sample
                        idx += 1

            #self.clean_data()
                self.samples = self.samples.values()
                self.save_samples(modes[i])

        else:
            self.samples = self.load_samples(mode)

            ### test meta data has missing annotation, only acquire images in 'train' folder ###
            #self.mode = 'train'

            self.sample_len = len(self.samples)
            self.subject_order = np.arange(self.sample_len)

            self.samples_reorder = dict()
            idx = 0
            for subject in self.subject_order:
                for k, v in self.samples[subject].items():
                    self.samples_reorder[idx] = v
                    idx += 1

            self.samples = self.samples_reorder
            self.sample_len = len(self.samples)


    def load_samples(self, mode):
        with open('../cfg/HO3D_v2/{}.pkl'.format(self.name), 'rb') as f:
            samples = pickle.load(f)
            return samples

    def save_samples(self, mode):
        with open('../cfg/HO3D_v2/{}.pkl'.format(self.name), 'wb') as f:
            pickle.dump(list(self.samples), f, pickle.HIGHEST_PROTOCOL)

    def clean_data(self):
        print("Size beforing cleaning: {}".format(len(self.samples.keys())))

        for key in list(self.samples):
            try:
                self.__getitem__(key)
            except Exception as e:
                print(e)
                print("Index failed: {}".format(key))
                del self.samples[key]

        self.samples = self.samples.values()

        print("Size after cleaning: {}".format(len(self.samples)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.preprocess(idx)

    def get_image(self, sample, flag_set):
        img = self.fetch_image(sample)
        img = self.transform(img)
        img = np.asarray(img.resize((416, 416), Image.ANTIALIAS), dtype=np.float32)
        if img.shape[-1] != 3:
            img = img[:, :, :-1]

        if flag_set[0]:
            img = np.flip(img, axis=0)
        if flag_set[1]:
            img = np.flip(img, axis=1)

        img = img / 255.
        # cv2.imshow("img in dataset", img)
        # cv2.waitKey(0)
        img = np.squeeze(np.transpose(img, (2, 0, 1)))

        return img

    # def get_depth(self, sample):
    #
    #     img = self.fetch_depth(sample)
    #     img_np = np.array(img)
    #
    #     if self.mode == 'train':
    #         img = self.transform(img)
    #     img = np.asarray(img.resize((416, 416), Image.ANTIALIAS), dtype=np.float32)
    #     #img = np.flip(img, axis=1)
    #     img = img / 255.
    #     # cv2.imshow("img in dataset", img)
    #     # cv2.waitKey(1)
    #     img = np.transpose(img, (2, 0, 1))
    #     return img
    #
    # def read_image(self, sample):
    #     file_name = sample['frame_idx'] + '.png'
    #     img_path = os.path.join(self.root, self.mode, sample['subject'], 'rgb', file_name)
    #
    #     img = cv2.imread(img_path)
    #     return img
    #
    def fetch_image(self, sample):
        file_name = sample['frame_idx'] + '.png'
        img_path = os.path.join(self.root, 'train', sample['subject'], 'rgb', file_name)
        _assert_exist(img_path)

        img = Image.open(img_path)
        return img
    #
    # def fetch_depth(self, sample):
    #     file_name = sample['frame_idx'] + '.png'
    #     img_path = os.path.join(self.root, self.mode, sample['subject'], 'depth', file_name)
    #     img = Image.open(img_path)
    #     return img

    def read_data(self, sample):

        file_name = sample['frame_idx'] + '.pkl'
        meta_path = os.path.join(self.root, 'train', sample['subject'], 'meta', file_name)
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)

        file_name = sample['frame_idx'] + '.png'
        img_path = os.path.join(self.root, 'train', sample['subject'], 'rgb', file_name)
        _assert_exist(img_path)
        rgb = cv2.imread(img_path)

        img_path = os.path.join(self.root, 'train', sample['subject'], 'depth', file_name)
        _assert_exist(img_path)
        depth_scale = 0.00012498664727900177
        depth = cv2.imread(img_path)

        dpt = depth[:, :, 2] + depth[:, :, 1] * 256
        dpt = dpt * depth_scale

        return rgb, dpt, meta


    def preprocess(self, idx):
        """
        objTrans: A 3x1 vector representing object translation
        objRot: A 3x1 vector representing object rotation in axis-angle representation
        handPose: A 48x1 vector represeting the 3D rotation of the 16 hand joints including the root joint in axis-angle representation. The ordering of the joints follow the MANO model convention (see joint_order.png) and can be directly fed to MANO model.
        handTrans: A 3x1 vector representing the hand translation
        handBeta: A 10x1 vector representing the MANO hand shape parameters
        handJoints3D: A 21x3 matrix representing the 21 3D hand joint locations
        objCorners3D: A 8x3 matrix representing the 3D bounding box corners of the object
        objCorners3DRest: A 8x3 matrix representing the 3D bounding box corners of the object before applying the transformation
        objName: Name of the object as given in YCB dataset
        objLabel: Object label as given in YCB dataset
        camMat: Intrinsic camera parameters
        """
        #idx = idx % (self.sample_len)
        sample = self.samples[idx]

        frame_idx = int(sample['frame_idx'])

        img, depth, meta = self.read_data(sample)
        # subject = sample['subject']

        objCorners = meta['objCorners3DRest']
        objCornersTrans = np.matmul(objCorners, cv2.Rodrigues(meta['objRot'])[0].T) + meta['objTrans']
        objCornersTrans = objCornersTrans.dot(self.coord_change_mat.T) * 1000.
        objcontrolPoints = self.get_box_3d_control_points(objCornersTrans)

        handJoints3D = meta['handJoints3D']
        handJoints3D = handJoints3D.dot(self.coord_change_mat.T) * 1000.
        handKps = project_3D_points(meta['camMat'], handJoints3D, is_OpenGL_coords=False)

        handKps_ = handKps[jointsMapManoToSimple]
        handJoints3D_ = handJoints3D[jointsMapManoToSimple]
        handKps_ = np.round(handKps_).astype(np.int)
        visible = []
        for i in range(21):
            if handKps_[i][0] >= 640 or handKps_[i][1] >= 480:
                continue
            d_img = depth[handKps_[i][1], handKps_[i][0]]
            d_gt = handJoints3D_[i][-1]
            if np.abs(d_img - d_gt) < VISIBLE_PARAM:
                visible.append(i)
        """
        # depth_proc = np.copy(depth)
        # depth_proc[depth > 1.0] = 0.0
        # depthAnno = showHandJoints(depth_proc, handKps)
        imgAnno = showHandJoints_vis(img, handKps, vis=visible)
        # imgAnno = showHandJoints(img, handKps)
        # imgAnno = showObjJoints(imgAnno, objKps, lineThickness=2)
        rgb = img[:, :, [0, 1, 2]]
        cv2.imshow("rgb", rgb)
        anno = imgAnno[:, :, [0, 1, 2]]
        cv2.imshow("anno", anno)
        # cv2.imshow("depthAnno", depthAnno)
        cv2.waitKey(1)

        handJoints3D_vert = np.zeros((21, 3), dtype=np.float64)
        handJoints3D_vert[:, 1] = handJoints3D[:, 1] * -1
        handJoints3D_vert[:, 0] = handJoints3D[:, 0]
        handJoints3D_vert[:, -1] = handJoints3D[:, -1]
        handKps_vert = project_3D_points(meta['camMat'], handJoints3D_vert, is_OpenGL_coords=False)

        img = np.flip(img, axis=0)
        imgAnno = showHandJoints_vis(img, handKps_vert, vis=visible)
        # imgAnno = showHandJoints(img, handKps)
        # imgAnno = showObjJoints(imgAnno, objKps, lineThickness=2)
        rgb = img[:, :, [0, 1, 2]]
        cv2.imshow("rgb_vert", rgb)
        anno = imgAnno[:, :, [0, 1, 2]]
        cv2.imshow("anno_vert", anno)
        cv2.waitKey(0)
        """
        ### data augmentation ###

        flag_vert = np.random.randint(2)
        flag_hori = np.random.randint(2)
        flag_set = [flag_vert, flag_hori]

        handJoints3D_flip = np.copy(handJoints3D)
        objcontrolPoints_flip = np.copy(objcontrolPoints)
        if flag_vert:
            handJoints3D_flip[:, 1] = handJoints3D[:, 1] * -1
            objcontrolPoints_flip[:, 1] = objcontrolPoints[:, 1] * -1

        if flag_hori:
            handJoints3D_flip[:, 0] = handJoints3D[:, 0] * -1
            objcontrolPoints_flip[:, 0] = objcontrolPoints[:, 0] * -1

        objcontrolPoints = objcontrolPoints_flip
        handJoints3D = handJoints3D_flip

        handKps = project_3D_points(meta['camMat'], handJoints3D, is_OpenGL_coords=False)
        handKps = handKps[jointsMapManoToSimple]
        handJoints3D = handJoints3D[jointsMapManoToSimple]

        objKps = project_3D_points(meta['camMat'], objcontrolPoints, is_OpenGL_coords=False)

        ### object pose ###
        # get offset w.r.t top/left corner of the cell
        del_u, del_v, del_z, cell = self.control_to_target(objKps, objcontrolPoints)

        # object pose tensor
        true_object_pose = torch.zeros(21, 3, 5, 13, 13, dtype=torch.float32)
        u, v, z = cell
        pose = np.vstack((del_u, del_v, del_z)).T
        true_object_pose[:, :, z, v, u] = torch.from_numpy(pose)
        true_object_pose = true_object_pose.view(-1, 5, 13, 13)     # (63, 5, 13, 13)

        # object mask
        object_mask = torch.zeros(5, 13, 13, dtype=torch.float32)
        object_mask[z, v, u] = 1


        ### hand pose ###
        del_u, del_v, del_z, cell = self.control_to_target(handKps, handJoints3D)  # handKps : [215.8, 182.1] , ...   /  handJoints3D : [~, ~, 462.2] , ...
        # hand pose tensor
        true_hand_pose = torch.zeros(21, 3, 5, 13, 13, dtype=torch.float32)
        u, v, z = cell
        pose = np.vstack((del_u, del_v, del_z)).T
        true_hand_pose[:, :, z, v, u] = torch.from_numpy(pose)
        true_hand_pose = true_hand_pose.view(-1, 5, 13, 13)
        # hand mask
        hand_mask = torch.zeros(5, 13, 13, dtype=torch.float32)
        hand_mask[z, v, u] = 1

        # hand visibility per joint
        vis = np.zeros(21, dtype=np.float32)
        if len(visible) != 0:
            vis[np.array(visible)] = 1

        param_vis = torch.zeros(21, 5, 13, 13, dtype=torch.float32)
        param_vis[:, z, v, u] = torch.from_numpy(vis)
        param_vis = param_vis.view(-1, 5, 13, 13)

        image = None

        if self.loadit:
            image = torch.from_numpy(self.get_image(sample, flag_set))
            if int(image.shape[0]) != 3:
                print("image shape wrong")
                image = image[:-1, :, :]
            image = self.normalize(image)

        ### preprocess extrapolated keypoint GT ###
        if self.mode == 'train':
            prev_handKps = dict()
            prev_handJoints3D = dict()
            prev_vis = dict()

            if np.abs(frame_idx - self.prev_frame_idx) > 2:
                self.counting_idx = 0

            self.prev_frame_idx = frame_idx

            if self.counting_idx < 2:
                self.counting_idx += 1

                for i in range(2):
                    prev_handKps[i] = np.zeros((21, 2), dtype=np.float32)
                    prev_handJoints3D[i] = np.zeros((21, 3), dtype=np.float32)
                    prev_vis[i] = np.zeros(21, dtype=np.float32)

            else:
                for i in range(2):
                    prev_sample = self.samples[idx - (i + 1)]
                    img, depth, meta = self.read_data(prev_sample)

                    handJoints3D = meta['handJoints3D']
                    handJoints3D = handJoints3D.dot(self.coord_change_mat.T) * 1000.

                    ##################################################################

                    handKps_ = project_3D_points(meta['camMat'], handJoints3D, is_OpenGL_coords=False)

                    handKps_ = handKps_[jointsMapManoToSimple]
                    handJoints3D_ = handJoints3D[jointsMapManoToSimple]
                    handKps_ = np.round(handKps_).astype(np.int)
                    visible = []
                    for j in range(21):
                        if handKps_[j][0] >= 640 or handKps_[j][1] >= 480:
                            continue
                        d_img = depth[handKps_[j][1], handKps_[j][0]]
                        d_gt = handJoints3D_[j][-1]
                        if np.abs(d_img - d_gt) < VISIBLE_PARAM:
                            visible.append(j)
                    vis = np.zeros(21, dtype=np.float32)
                    if len(visible) != 0:
                        vis[np.array(visible)] = 1

                    ##################################################################

                    handJoints3D_flip = np.copy(handJoints3D)
                    if flag_vert:
                        handJoints3D_flip[:, 1] = handJoints3D[:, 1] * -1
                    if flag_hori:
                        handJoints3D_flip[:, 0] = handJoints3D[:, 0] * -1

                    handJoints3D = handJoints3D_flip

                    handKps = project_3D_points(meta['camMat'], handJoints3D, is_OpenGL_coords=False)

                    handKps = handKps[jointsMapManoToSimple]
                    handJoints3D = handJoints3D[jointsMapManoToSimple]

                    prev_handKps[i] = handKps
                    prev_handJoints3D[i] = handJoints3D
                    prev_vis[i] = vis[jointsMapManoToSimple]

            extra_handKps = 2 * prev_handKps[0] - prev_handKps[1]
            extra_handJoints3D = 2 * prev_handJoints3D[0] - prev_handJoints3D[1]

            mixed_handKps = np.copy(prev_handKps[0])
            mixed_handJoints3D = np.copy(prev_handJoints3D[0])

            visible_index = np.multiply(prev_vis[0], prev_vis[1]).astype(int)

            mixed_handKps[visible_index, :] = extra_handKps[visible_index, :]
            mixed_handJoints3D[visible_index, :] = extra_handJoints3D[visible_index, :]

            del_u, del_v, del_z, cell = self.control_to_target(mixed_handKps, mixed_handJoints3D)

            # hand pose tensor
            # index + del, with positional encoding
            del_u = torch.unsqueeze(torch.from_numpy(del_u), 0).type(torch.float32)
            del_v = torch.unsqueeze(torch.from_numpy(del_v), 0).type(torch.float32)
            del_z = torch.unsqueeze(torch.from_numpy(del_z), 0).type(torch.float32)

            enc_cell = self.pos_encoder[:, cell[0], cell[1], cell[2], :].type(torch.float32)

            extra_hand_pose = torch.cat((enc_cell, del_u, del_v, del_z), 0) # tensor (4, 21)

            return image, true_hand_pose, hand_mask, true_object_pose, object_mask, param_vis, vis, extra_hand_pose #, flag_set

        # for test code
        else:
            if np.abs(frame_idx - self.prev_frame_idx) > 2:
                flag_seq = True
            else:
                flag_seq = False

            self.prev_frame_idx = frame_idx

            return image, true_hand_pose, hand_mask, true_object_pose, object_mask, param_vis, vis, flag_seq


    def load_objects(self, obj_root):
        object_names = ['juice', 'liquid_soap', 'milk', 'salt']
        all_models = {}
        for obj_name in object_names:
            obj_path = os.path.join(obj_root, '{}_model'.format(obj_name),
                                    '{}_model.ply'.format(obj_name))
            mesh = trimesh.load(obj_path)
            corners = trimesh.bounds.corners(mesh.bounding_box.bounds)
            all_models[obj_name] = {
                'corners': corners
            }
        return all_models

    def get_skeleton(self, sample, skel_root):
        skeleton_path = os.path.join(skel_root, sample['subject'],
                                     sample['action_name'], sample['seq_idx'],
                                     'skeleton.txt')
        #print('Loading skeleton from {}'.format(skeleton_path))
        skeleton_vals = np.loadtxt(skeleton_path)
        skeleton = skeleton_vals[:, 1:].reshape(skeleton_vals.shape[0], 21,
                                                -1)[sample['frame_idx']]
        return skeleton

    def get_object_pose(self, sample, obj_root):
        seq_path = os.path.join(obj_root, sample['subject'], sample['action_name'],
                                sample['seq_idx'], 'object_pose.txt')
        with open(seq_path, 'r') as seq_f:
            raw_lines = seq_f.readlines()
        raw_line = raw_lines[sample['frame_idx']]
        line = raw_line.strip().split(' ')
        trans_matrix = np.array(line[1:]).astype(np.float32)
        trans_matrix = trans_matrix.reshape(4, 4).transpose()
        # print('Loading obj transform from {}'.format(seq_path))
        return trans_matrix

    def downsample_points(self, points, depth):

        downsample_ratio_x = 640 / 416.
        downsample_ratio_y = 480 / 416.

        x = points[0] / downsample_ratio_x
        y = points[1] / downsample_ratio_y
        z = depth / 10.  # converting to centimeters

        downsampled_x = x / 32
        downsampled_y = y / 32
        downsampled_z = z / 15

        return downsampled_x, downsampled_y, downsampled_z

    def upsample_points(self, points, depth):

        downsample_ratio_x = 640 / 416.
        downsample_ratio_y = 480 / 416.

        u = points[0] * downsample_ratio_x
        v = points[1] * downsample_ratio_y
        z = depth * 10.  # converting to millimeters

        return u, v, z

    def get_cell(self, root, depth):

        downsampled_x, downsampled_y, downsampled_z = self.downsample_points(root, depth)

        u = int(downsampled_x)
        v = int(downsampled_y)
        z = int(downsampled_z)

        return (u, v, z)

    def compute_offset(self, points, cell):

        points_u, points_v, points_z = points
        points_u, points_v, points_z = self.downsample_points((points_u, points_v), points_z)
        cell_u, cell_v, cell_z = cell
        del_u = points_u - cell_u
        del_v = points_v - cell_v
        del_z = points_z - cell_z

        return del_u, del_v, del_z

    def get_box_3d_control_points(self, corners):

        # lines (0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4), (0,4), (1,5), (2,6), (3,7)

        edge_01 = (corners[0] + corners[1]) / 2.
        edge_12 = (corners[1] + corners[2]) / 2.
        edge_23 = (corners[2] + corners[3]) / 2.
        edge_30 = (corners[3] + corners[0]) / 2.
        edge_45 = (corners[4] + corners[5]) / 2.
        edge_56 = (corners[5] + corners[6]) / 2.
        edge_67 = (corners[6] + corners[7]) / 2.
        edge_74 = (corners[7] + corners[4]) / 2.
        edge_04 = (corners[0] + corners[4]) / 2.
        edge_15 = (corners[1] + corners[5]) / 2.
        edge_26 = (corners[2] + corners[6]) / 2.
        edge_37 = (corners[3] + corners[7]) / 2.

        center = np.mean(corners, axis=0)

        control_points = np.vstack((center, corners,
                                    edge_01, edge_12, edge_23, edge_30,
                                    edge_45, edge_56, edge_67, edge_74,
                                    edge_04, edge_15, edge_26, edge_37))

        return control_points

    def control_to_target(self, projected_points, points):
        # 2D location : projected_points[0, :]
        # depth : points[0, 2]
        root = projected_points[0, :]

        # get location of cell in (13, 13, 5)
        cell = self.get_cell(root, points[0, 2])
        u, v, z = cell
        if u > 12:
            u = 12
        if v > 12:
            v = 12
        if z > 4:
            z = 4
        cell = [u, v, z]

        points = projected_points[:, 0], projected_points[:, 1], points[:, 2]  # px, px, mm

        del_u, del_v, del_z = self.compute_offset(points, cell)

        return del_u, del_v, del_z, cell

    def target_to_control(self, del_u, del_v, del_z, cell):

        u, v, z = cell

        w_u = del_u + u
        w_v = del_v + v
        w_z = del_z + z

        w_u, w_v, w_z = self.upsample_points((w_u, w_v), w_z)
        points = np.vstack((w_u * 32, w_v * 32, np.ones_like(w_u)))
        y_hat = w_z * 15 * np.linalg.inv(self.camera_intrinsics).dot(points)

        return y_hat.T, points


class HO3D_v2_Dataset(Dataset):

    def __init__(self, mode='train', root='../../dataset/HO3D_V2', cfg='train',loadit=False, shuffle_seq=False):
        ###
        # initial setting
        # 640*480 image to 32*32*5 box
        # only control depth_discretization parameter(=5) in cfg.yaml
        ###
        self.coord_change_mat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)

        self.name = cfg
        self.root = root
        self.loadit = loadit
        self.mode = mode

        self.transform = transforms.ColorJitter(0.5, 0.5, 0.5)
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        # load meshes to memory
        # object_root = os.path.join(self.root, 'Object_models')
        # self.objects = self.load_objects(object_root)

        # self.camera_pose = np.array(
        #     [[0.999988496304, -0.00468848412856, 0.000982563360594, 25.7],
        #      [0.00469115935266, 0.999985218048, -0.00273845880292, 1.22],
        #      [-0.000969709653873, 0.00274303671904, 0.99999576807, 3.902],
        #      [0, 0, 0, 1]])

        self.camera_pose = np.array(
            [[1., 0., 0., 0.],
             [0., 1., 0., 0.],
             [0., 0., 1., 0.],
             [0, 0, 0, 1]])

        self.camera_intrinsics = np.array([[1395.749023, 0, 935.732544],
                                           [0, 1395.749268, 540.681030],
                                           [0, 0, 1]])

        self.depth_intrinsics = np.array([[475.065948, 0, 315.944855],
                                           [0,  475.065857, 245.287079],
                                           [0, 0, 1]])

        if not loadit:

            # subjects = [1, 2, 3, 4, 5, 6]
            # subject = "Subject_1"
            # subject = os.path.join(root, 'Object_6D_pose_annotation_v1', subject)
            # actions = os.listdir(subject)

            subject_path = os.path.join(root, mode)
            subjects = os.listdir(subject_path)

            dataset = dict()
            dataset['train'] = dict()
            dataset['test'] = dict()

            for subject in subjects:
                subject = str(subject)

                dataset['train'][subject] = list()
                dataset['test'][subject] = list()

                rgb_set = list(os.listdir(os.path.join(root, mode, subject, 'rgb')))
                frames = len(rgb_set)
                # random.shuffle(rgb_set)

                data_split = int(frames * 4 / 5) + 1

                for i in range(frames):
                    if i < data_split:
                        dataset['train'][subject].append(rgb_set[i])
                    else:
                        dataset['test'][subject].append(rgb_set[i])

            print(yaml.dump(dataset))

            modes = ['train', 'test']
            for i in range(2):
                self.samples = dict()
                self.name = modes[i]

                for subject in list(dataset[modes[i]]):
                    self.samples[subject] = dict()

                    idx = 0
                    for frame in dataset[modes[i]][subject]:
                        sample = {
                            'subject': subject,
                            'frame_idx': frame[:-4],
                        }
                        self.samples[subject][idx] = sample
                        idx += 1

            #self.clean_data()
                self.samples = self.samples.values()
                self.save_samples(modes[i])

        else:
            self.samples = self.load_samples(mode)
            ### test meta data has missing annotation, split training dataset and test
            self.mode = 'train'


            self.sample_len = len(self.samples)
            self.subject_order = np.arange(self.sample_len)
            if shuffle_seq:
                np.random.shuffle(self.subject_order)
                sub_len = int(len(self.subject_order) / 4)
                self.subject_order = self.subject_order[:sub_len]

            self.samples_reorder = dict()
            idx = 0
            for subject in self.subject_order:
                for k, v in self.samples[subject].items():
                    self.samples_reorder[idx] = v
                    idx += 1

            self.samples = self.samples_reorder
            self.sample_len = len(self.samples)


    def load_samples(self, mode):
        with open('../cfg/HO3D_v2/{}.pkl'.format(self.name), 'rb') as f:
            samples = pickle.load(f)
            return samples

    def save_samples(self, mode):
        with open('../cfg/HO3D_v2/{}.pkl'.format(self.name), 'wb') as f:
            pickle.dump(list(self.samples), f, pickle.HIGHEST_PROTOCOL)

    def clean_data(self):
        print("Size beforing cleaning: {}".format(len(self.samples.keys())))

        for key in list(self.samples):
            try:
                self.__getitem__(key)
            except Exception as e:
                print(e)
                print("Index failed: {}".format(key))
                del self.samples[key]

        self.samples = self.samples.values()

        print("Size after cleaning: {}".format(len(self.samples)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.preprocess(idx)

    def get_image(self, sample):

        img = self.fetch_image(sample)

        if self.mode == 'train':
            img = self.transform(img)
        img = np.asarray(img.resize((416, 416), Image.ANTIALIAS), dtype=np.float32)
        if img.shape[-1] != 3:
            img = img[:, :, :-1]
        #img = np.flip(img, axis=1)
        img = img / 255.
        # cv2.imshow("img in dataset", img)
        # cv2.waitKey(1)
        img = np.squeeze(np.transpose(img, (2, 0, 1)))
        return img

    # def get_depth(self, sample):
    #
    #     img = self.fetch_depth(sample)
    #     img_np = np.array(img)
    #
    #     if self.mode == 'train':
    #         img = self.transform(img)
    #     img = np.asarray(img.resize((416, 416), Image.ANTIALIAS), dtype=np.float32)
    #     #img = np.flip(img, axis=1)
    #     img = img / 255.
    #     # cv2.imshow("img in dataset", img)
    #     # cv2.waitKey(1)
    #     img = np.transpose(img, (2, 0, 1))
    #     return img
    #
    # def read_image(self, sample):
    #     file_name = sample['frame_idx'] + '.png'
    #     img_path = os.path.join(self.root, self.mode, sample['subject'], 'rgb', file_name)
    #
    #     img = cv2.imread(img_path)
    #     return img
    #
    def fetch_image(self, sample):
        file_name = sample['frame_idx'] + '.png'
        img_path = os.path.join(self.root, self.mode, sample['subject'], 'rgb', file_name)
        _assert_exist(img_path)

        img = Image.open(img_path)
        return img
    #
    # def fetch_depth(self, sample):
    #     file_name = sample['frame_idx'] + '.png'
    #     img_path = os.path.join(self.root, self.mode, sample['subject'], 'depth', file_name)
    #     img = Image.open(img_path)
    #     return img

    def read_data(self, sample):

        file_name = sample['frame_idx'] + '.pkl'
        meta_path = os.path.join(self.root, self.mode, sample['subject'], 'meta', file_name)
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)

        file_name = sample['frame_idx'] + '.png'
        img_path = os.path.join(self.root, self.mode, sample['subject'], 'rgb', file_name)
        _assert_exist(img_path)
        rgb = cv2.imread(img_path)

        img_path = os.path.join(self.root, self.mode, sample['subject'], 'depth', file_name)
        _assert_exist(img_path)
        depth_scale = 0.00012498664727900177
        depth = cv2.imread(img_path)

        dpt = depth[:, :, 2] + depth[:, :, 1] * 256
        dpt = dpt * depth_scale

        return rgb, dpt, meta


    def preprocess(self, idx):
        """
        objTrans: A 3x1 vector representing object translation
        objRot: A 3x1 vector representing object rotation in axis-angle representation
        handPose: A 48x1 vector represeting the 3D rotation of the 16 hand joints including the root joint in axis-angle representation. The ordering of the joints follow the MANO model convention (see joint_order.png) and can be directly fed to MANO model.
        handTrans: A 3x1 vector representing the hand translation
        handBeta: A 10x1 vector representing the MANO hand shape parameters
        handJoints3D: A 21x3 matrix representing the 21 3D hand joint locations
        objCorners3D: A 8x3 matrix representing the 3D bounding box corners of the object
        objCorners3DRest: A 8x3 matrix representing the 3D bounding box corners of the object before applying the transformation
        objName: Name of the object as given in YCB dataset
        objLabel: Object label as given in YCB dataset
        camMat: Intrinsic camera parameters
        """
        #idx = idx % (self.sample_len)
        sample = self.samples[idx]
        img, depth, meta = self.read_data(sample)
        subject = sample['subject']

        depth_proc = np.copy(depth)
        depth_proc[depth > 1.0] = 0.0

        objCorners = meta['objCorners3DRest']
        objCornersTrans = np.matmul(objCorners, cv2.Rodrigues(meta['objRot'])[0].T) + meta['objTrans']
        objCornersTrans = objCornersTrans.dot(self.coord_change_mat.T) * 1000.
        objcontrolPoints = self.get_box_3d_control_points(objCornersTrans)
        objKps = project_3D_points(meta['camMat'], objcontrolPoints, is_OpenGL_coords=False)

        handJoints3D = meta['handJoints3D']
        handJoints3D = handJoints3D.dot(self.coord_change_mat.T) # * 1000.
        handKps = project_3D_points(meta['camMat'], handJoints3D, is_OpenGL_coords=False)
        handKps = handKps[jointsMapManoToSimple]

        handJoints3D_ = handJoints3D[jointsMapManoToSimple]
        handKps_ = np.round(handKps).astype(np.int)

        visible = []
        for i in range(21):
            if handKps_[i][0] >= 640 or handKps_[i][1] >= 480:
                continue
            d_img = depth[handKps_[i][1], handKps_[i][0]]
            d_gt = handJoints3D_[i][-1]
            if np.abs(d_img - d_gt) < VISIBLE_PARAM:
                visible.append(i)


        # imgAnno = showHandJoints_vis(img, handKps, vis=visible)
        # depthAnno = showHandJoints(depth_proc, handKps)
        #
        # imgAnno = showHandJoints(img, handKps)
        # # imgAnno = showObjJoints(imgAnno, objKps, lineThickness=2)
        # rgb = img[:, :, [0, 1, 2]]
        # cv2.imshow("rgb", rgb)
        # anno = imgAnno[:, :, [0, 1, 2]]
        # cv2.imshow("anno", anno)
        # # cv2.imshow("depthAnno", depthAnno)
        # cv2.waitKey(0)

        # get offset w.r.t top/left corner of the cell
        del_u, del_v, del_z, cell = self.control_to_target(objKps, objcontrolPoints)

        # object pose tensor
        true_object_pose = torch.zeros(21, 3, 5, 13, 13, dtype=torch.float32)
        u, v, z = cell
        if u > 12:
            u = 12
        if v > 12:
            v = 12
        if z > 4:
            z = 4
        pose = np.vstack((del_u, del_v, del_z)).T
        true_object_pose[:, :, z, v, u] = torch.from_numpy(pose)
        true_object_pose = true_object_pose.view(-1, 5, 13, 13)     # (63, 5, 13, 13)

        # object mask
        object_mask = torch.zeros(5, 13, 13, dtype=torch.float32)
        object_mask[z, v, u] = 1

        del_u, del_v, del_z, cell = self.control_to_target(handKps, handJoints3D)

        # hand pose tensor
        true_hand_pose = torch.zeros(21, 3, 5, 13, 13, dtype=torch.float32)
        u, v, z = cell
        pose = np.vstack((del_u, del_v, del_z)).T
        if u > 12:
            u = 12
        if v > 12:
            v = 12
        if z > 4:
            z = 4
        true_hand_pose[:, :, z, v, u] = torch.from_numpy(pose)
        true_hand_pose = true_hand_pose.view(-1, 5, 13, 13)

        # hand mask
        hand_mask = torch.zeros(5, 13, 13, dtype=torch.float32)
        hand_mask[z, v, u] = 1

        # hand visibility per joint
        vis = np.zeros(21, dtype=np.float32)
        if len(visible) != 0:
            vis[np.array(visible)] = 1

        param_vis = torch.zeros(21, 5, 13, 13, dtype=torch.float32)
        param_vis[:, z, v, u] = torch.from_numpy(vis)
        param_vis = param_vis.view(-1, 5, 13, 13)

        image = None

        if self.loadit:
            image = torch.from_numpy(self.get_image(sample))
            if int(image.shape[0]) != 3:
                print("image shpae wrong")
                image = image[:-1, :, :]
            image = self.normalize(image)

        return image, true_hand_pose, hand_mask, true_object_pose, object_mask, param_vis, subject, vis

    def load_objects(self, obj_root):
        object_names = ['juice', 'liquid_soap', 'milk', 'salt']
        all_models = {}
        for obj_name in object_names:
            obj_path = os.path.join(obj_root, '{}_model'.format(obj_name),
                                    '{}_model.ply'.format(obj_name))
            mesh = trimesh.load(obj_path)
            corners = trimesh.bounds.corners(mesh.bounding_box.bounds)
            all_models[obj_name] = {
                'corners': corners
            }
        return all_models

    def get_skeleton(self, sample, skel_root):
        skeleton_path = os.path.join(skel_root, sample['subject'],
                                     sample['action_name'], sample['seq_idx'],
                                     'skeleton.txt')
        #print('Loading skeleton from {}'.format(skeleton_path))
        skeleton_vals = np.loadtxt(skeleton_path)
        skeleton = skeleton_vals[:, 1:].reshape(skeleton_vals.shape[0], 21,
                                                -1)[sample['frame_idx']]
        return skeleton

    def get_object_pose(self, sample, obj_root):
        seq_path = os.path.join(obj_root, sample['subject'], sample['action_name'],
                                sample['seq_idx'], 'object_pose.txt')
        with open(seq_path, 'r') as seq_f:
            raw_lines = seq_f.readlines()
        raw_line = raw_lines[sample['frame_idx']]
        line = raw_line.strip().split(' ')
        trans_matrix = np.array(line[1:]).astype(np.float32)
        trans_matrix = trans_matrix.reshape(4, 4).transpose()
        # print('Loading obj transform from {}'.format(seq_path))
        return trans_matrix

    def downsample_points(self, points, depth):

        downsample_ratio_x = 640 / 416.
        downsample_ratio_y = 480 / 416.

        x = points[0] / downsample_ratio_x
        y = points[1] / downsample_ratio_y
        z = depth / 10.  # converting to centimeters

        downsampled_x = x / 32
        downsampled_y = y / 32
        downsampled_z = z / 15

        return downsampled_x, downsampled_y, downsampled_z

    def upsample_points(self, points, depth):

        downsample_ratio_x = 640 / 416.
        downsample_ratio_y = 480 / 416.

        u = points[0] * downsample_ratio_x
        v = points[1] * downsample_ratio_y
        z = depth * 10.  # converting to millimeters

        return u, v, z

    def get_cell(self, root, depth):

        downsampled_x, downsampled_y, downsampled_z = self.downsample_points(root, depth)

        u = int(downsampled_x)
        v = int(downsampled_y)
        z = int(downsampled_z)

        return (u, v, z)

    def compute_offset(self, points, cell):

        points_u, points_v, points_z = points
        points_u, points_v, points_z = self.downsample_points((points_u, points_v), points_z)
        cell_u, cell_v, cell_z = cell
        del_u = points_u - cell_u
        del_v = points_v - cell_v
        del_z = points_z - cell_z

        return del_u, del_v, del_z

    def get_box_3d_control_points(self, corners):

        # lines (0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4), (0,4), (1,5), (2,6), (3,7)

        edge_01 = (corners[0] + corners[1]) / 2.
        edge_12 = (corners[1] + corners[2]) / 2.
        edge_23 = (corners[2] + corners[3]) / 2.
        edge_30 = (corners[3] + corners[0]) / 2.
        edge_45 = (corners[4] + corners[5]) / 2.
        edge_56 = (corners[5] + corners[6]) / 2.
        edge_67 = (corners[6] + corners[7]) / 2.
        edge_74 = (corners[7] + corners[4]) / 2.
        edge_04 = (corners[0] + corners[4]) / 2.
        edge_15 = (corners[1] + corners[5]) / 2.
        edge_26 = (corners[2] + corners[6]) / 2.
        edge_37 = (corners[3] + corners[7]) / 2.

        center = np.mean(corners, axis=0)

        control_points = np.vstack((center, corners,
                                    edge_01, edge_12, edge_23, edge_30,
                                    edge_45, edge_56, edge_67, edge_74,
                                    edge_04, edge_15, edge_26, edge_37))

        return control_points

    def control_to_target(self, projected_points, points):
        # 2D location : projected_points[0, :]
        # depth : points[0, 2]
        root = projected_points[0, :]

        # get location of cell in (13, 13, 5)
        cell = self.get_cell(root, points[0, 2])

        points = projected_points[:, 0], projected_points[:, 1], points[:, 2]  # px, px, mm

        del_u, del_v, del_z = self.compute_offset(points, cell)

        return del_u, del_v, del_z, cell

    def target_to_control(self, del_u, del_v, del_z, cell):

        u, v, z = cell

        w_u = del_u + u
        w_v = del_v + v
        w_z = del_z + z

        w_u, w_v, w_z = self.upsample_points((w_u, w_v), w_z)
        points = np.vstack((w_u * 32, w_v * 32, np.ones_like(w_u)))
        y_hat = w_z * 15 * np.linalg.inv(self.camera_intrinsics).dot(points)

        return y_hat.T, points


class UnifiedPoseDataset(Dataset):

    def __init__(self, mode='train', root='../../dataset/First_Person_Action_Benchmark-selected', loadit=False, name=None):
        ###
        # initial setting
        # 1920*1080 image to 32*32*5 box
        # only control depth_discretization parameter(=5) in cfg.yaml
        ###
        self.offset = 0

        self.name = name
        self.root = root
        self.loadit = loadit
        self.mode = mode

        self.transform = transforms.ColorJitter(0.5, 0.5, 0.5)
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        # load meshes to memory
        object_root = os.path.join(self.root, 'Object_models')
        self.objects = self.load_objects(object_root)

        self.camera_pose = np.array(
            [[0.999988496304, -0.00468848412856, 0.000982563360594, 25.7],
             [0.00469115935266, 0.999985218048, -0.00273845880292, 1.22],
             [-0.000969709653873, 0.00274303671904, 0.99999576807, 3.902],
             [0, 0, 0, 1]])

        self.camera_intrinsics = np.array([[1395.749023, 0, 935.732544],
                                           [0, 1395.749268, 540.681030],
                                           [0, 0, 1]])

        self.depth_intrinsics = np.array([[475.065948, 0, 315.944855],
                                           [0,  475.065857, 245.287079],
                                           [0, 0, 1]])

        if not loadit:

            subjects = [1, 2, 3, 4, 5, 6]
            subject = "Subject_1"
            subject = os.path.join(root, 'Object_6D_pose_annotation_v1', subject)
            actions = os.listdir(subject)

            action_to_object = {
                'open_milk': 'milk',
                'close_milk': 'milk',
                'pour_milk': 'milk',
                'open_juice_bottle': 'juice',
                'close_juice_bottle': 'juice',
                'pour_juice_bottle': 'juice',
                'open_liquid_soap': 'liquid_soap',
                'close_liquid_soap': 'liquid_soap',
                'pour_liquid_soap': 'liquid_soap',
                'put_salt': 'salt'
            }

            idx = 0
            dataset = dict()
            dataset['train'] = dict()
            dataset['test'] = dict()

            for subject in subjects:
                subject = "Subject_" + str(subject)

                dataset['train'][subject] = dict()
                dataset['test'][subject] = dict()

                random.shuffle(actions)

                for action in actions:
                    dataset['train'][subject][action] = dict()
                    dataset['test'][subject][action] = dict()

                    pose_sequences = set(
                        os.listdir(os.path.join(root, 'Object_6D_pose_annotation_v1', subject, action)))
                    video_sequences = set(os.listdir(os.path.join(root, 'Video_files', subject, action)))
                    sequences = list(pose_sequences.intersection(video_sequences))

                    data_split = int(len(sequences) * 3 / 4) + 1

                    random.shuffle(sequences)

                    for sequence in sequences:

                        if int(sequence) < data_split:
                            dataset['train'][subject][action][int(sequence)] = list()
                        else:
                            dataset['test'][subject][action][int(sequence)] = list()

                        frames = len(os.listdir(os.path.join(root, 'Video_files', subject, action, sequence, 'color')))
                        #frames_d = len(os.listdir(os.path.join(root, 'Video_files', subject, action, sequence, 'depth')))

                        for frame in range(frames):
                            if int(sequence) < data_split:
                                dataset['train'][subject][action][int(sequence)].append(frame)
                            else:
                                dataset['test'][subject][action][int(sequence)].append(frame)


            print(yaml.dump(dataset))
            self.samples = dict()
            idx = 0

            for subject in list(dataset[mode]):
                for action in list(dataset[mode][subject]):
                    for sequence in list(dataset[mode][subject][action]):
                        for frame in dataset[mode][subject][action][sequence]:
                            sample = {
                                'subject': subject,
                                'action_name': action,
                                'seq_idx': str(sequence),
                                'frame_idx': frame,
                                'object': action_to_object[action]
                            }
                            self.samples[idx] = sample
                            idx += 1

            self.clean_data()

            self.save_samples(mode)

        else:
            self.samples = self.load_samples(mode)

        self.sample_len = len(self.samples)
        self.randomize_order()

    def randomize_order(self):
        self.offset = random.randint(0, self.sample_len - 1)

    def load_samples(self, mode):
        with open('../cfg/{}.pkl'.format(self.name), 'rb') as f:
            samples = pickle.load(f)
            return samples

    def save_samples(self, mode):

        with open('../cfg/{}.pkl'.format(self.name), 'wb') as f:
            pickle.dump(list(self.samples), f, pickle.HIGHEST_PROTOCOL)

    def clean_data(self):
        print("Size beforing cleaning: {}".format(len(self.samples.keys())))

        for key in list(self.samples):
            try:
                self.__getitem__(key)
            except Exception as e:
                print(e)
                print("Index failed: {}".format(key))
                del self.samples[key]

        self.samples = self.samples.values()

        print("Size after cleaning: {}".format(len(self.samples)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        return self.preprocess(idx)

    def get_image(self, sample):

        img = self.fetch_image(sample)

        if self.mode == 'train':
            img = self.transform(img)
        img = np.asarray(img.resize((416, 416), Image.ANTIALIAS), dtype=np.float32)
        #img = np.flip(img, axis=1)
        img = img / 255.
        # cv2.imshow("img in dataset", img)
        # cv2.waitKey(1)
        img = np.transpose(img, (2, 0, 1))
        return img

    def get_depth(self, sample):

        img = self.fetch_depth(sample)
        img_np = np.array(img)

        if self.mode == 'train':
            img = self.transform(img)
        img = np.asarray(img.resize((416, 416), Image.ANTIALIAS), dtype=np.float32)
        #img = np.flip(img, axis=1)
        img = img / 255.
        # cv2.imshow("img in dataset", img)
        # cv2.waitKey(1)
        img = np.transpose(img, (2, 0, 1))
        return img

    def fetch_image(self, sample):
        img_path = os.path.join(self.root, 'Video_files', sample['subject'],
                                sample['action_name'], sample['seq_idx'], 'color',
                                'color_{:04d}.jpeg'.format(sample['frame_idx']))
        img = Image.open(img_path)
        return img

    def fetch_depth(self, sample):
        img_path = os.path.join(self.root, 'Video_files', sample['subject'],
                                sample['action_name'], sample['seq_idx'], 'depth',
                                'depth_{:04d}.png'.format(sample['frame_idx']))
        img = Image.open(img_path)
        return img

    def preprocess(self, idx):
        idx = (idx + self.offset) % (self.sample_len)

        sample = self.samples[idx]

        object_category = {
            'juice': 0,
            'liquid_soap': 1,
            'milk': 2,
            'salt': 3
        }

        action_category = {
            'open_milk': 0,
            'close_milk': 1,
            'pour_milk': 2,
            'open_juice_bottle': 0,
            'close_juice_bottle': 1,
            'pour_juice_bottle': 2,
            'open_liquid_soap': 0,
            'close_liquid_soap': 1,
            'pour_liquid_soap': 2,
            'put_salt': 3
        }

        skeleton_root = os.path.join(self.root, 'Hand_pose_annotation_v1')
        object_pose_root = os.path.join(self.root, 'Object_6D_pose_annotation_v1')

        # Object Properties
        ###
        # get FHAD object pose as 4*4 matrix
        # make homogeneous corner matrix with given object model's corner info.
        # rotate model's corner position with given object pose, and rotate again w.r.t camera pose
        # corner : (8, 3)
        # control_points : (21, 3) ~ (center point, corner points(8), edge vectors(12))
        ###

        object_pose = self.get_object_pose(sample, object_pose_root)
        corners = self.objects[sample['object']]['corners'] * 1000.
        homogeneous_corners = np.concatenate([corners, np.ones([corners.shape[0], 1])], axis=1)
        corners = object_pose.dot(homogeneous_corners.T).T
        corners = self.camera_pose.dot(corners.T).T[:, :3]
        control_points = self.get_box_3d_control_points(corners)
        homogeneous_control_points = np.array(self.camera_intrinsics).dot(control_points.T).T
        box_projection = (homogeneous_control_points / homogeneous_control_points[:, 2:])[:, :2]

        # get offset w.r.t top/left corner of the cell
        del_u, del_v, del_z, cell = self.control_to_target(box_projection, control_points)

        # object pose tensor
        true_object_pose = torch.zeros(21, 3, 5, 13, 13, dtype=torch.float32)
        u, v, z = cell
        pose = np.vstack((del_u, del_v, del_z)).T
        true_object_pose[:, :, z, v, u] = torch.from_numpy(pose)
        true_object_pose = true_object_pose.view(-1, 5, 13, 13)     # (63, 5, 13, 13)

        # object mask
        object_mask = torch.zeros(5, 13, 13, dtype=torch.float32)
        object_mask[z, v, u] = 1

        # object class tensor
        true_object_prob = torch.zeros(5, 13, 13, dtype=torch.long)
        true_object_prob[z, v, u] = object_category[sample['object']]

        ### depth image processing ###
        # depth = torch.from_numpy(self.get_depth(sample))
        #
        # # need (n, 3) array of points
        # depth_points = depth
        # homogeneous_depth = np.concatenate([depth_points, np.ones([depth_points.shape[0], 1])], 1)
        #
        # Depth = np.linalg.inv(self.camera_intrinsics).dot(depth_points)
        # DepthToRGB = self.camera_pose.dot(Depth.T).T[:, :3].astype(np.float32)
        # Depth_projection = (DepthToRGB / DepthToRGB[:, 2:])[:, :2]
        # ###

        # Hand Properties
        reorder_idx = np.array([0, 1, 6, 7, 8, 2, 9, 10, 11, 3, 12, 13, 14, 4, 15, 16, 17, 5, 18, 19, 20])
        skeleton = self.get_skeleton(sample, skeleton_root)[reorder_idx]
        homogeneous_skeleton = np.concatenate([skeleton, np.ones([skeleton.shape[0], 1])], 1)
        skeleton = self.camera_pose.dot(homogeneous_skeleton.T).T[:, :3].astype(np.float32)  # mm
        homogeneous_skeleton = np.array(self.camera_intrinsics).dot(skeleton.T).T
        skeleton_projection = (homogeneous_skeleton / homogeneous_skeleton[:, 2:])[:, :2]

        del_u, del_v, del_z, cell = self.control_to_target(skeleton_projection, skeleton)

        debug_skeleton = skeleton_projection
        # hand pose tensor
        true_hand_pose = torch.zeros(21, 3, 5, 13, 13, dtype=torch.float32)
        u, v, z = cell
        pose = np.vstack((del_u, del_v, del_z)).T
        true_hand_pose[:, :, z, v, u] = torch.from_numpy(pose)
        true_hand_pose = true_hand_pose.view(-1, 5, 13, 13)

        # hand mask
        hand_mask = torch.zeros(5, 13, 13, dtype=torch.float32)
        hand_mask[z, v, u] = 1

        # hand action tensor
        true_hand_prob = torch.zeros(5, 13, 13, dtype=torch.long)
        true_hand_prob[z, v, u] = action_category[sample['action_name']]

        image = None

        if self.loadit:
            image = torch.from_numpy(self.get_image(sample))
            image = self.normalize(image)

        seq_idx = sample['seq_idx']

        return image, true_hand_pose, true_hand_prob, hand_mask, true_object_pose, true_object_prob, object_mask, seq_idx#, debug_skeleton

    def load_objects(self, obj_root):
        object_names = ['juice', 'liquid_soap', 'milk', 'salt']
        all_models = {}
        for obj_name in object_names:
            obj_path = os.path.join(obj_root, '{}_model'.format(obj_name),
                                    '{}_model.ply'.format(obj_name))
            mesh = trimesh.load(obj_path)
            corners = trimesh.bounds.corners(mesh.bounding_box.bounds)
            all_models[obj_name] = {
                'corners': corners
            }
        return all_models

    def get_skeleton(self, sample, skel_root):
        skeleton_path = os.path.join(skel_root, sample['subject'],
                                     sample['action_name'], sample['seq_idx'],
                                     'skeleton.txt')
        #print('Loading skeleton from {}'.format(skeleton_path))
        skeleton_vals = np.loadtxt(skeleton_path)
        skeleton = skeleton_vals[:, 1:].reshape(skeleton_vals.shape[0], 21,
                                                -1)[sample['frame_idx']]
        return skeleton

    def get_object_pose(self, sample, obj_root):
        seq_path = os.path.join(obj_root, sample['subject'], sample['action_name'],
                                sample['seq_idx'], 'object_pose.txt')
        with open(seq_path, 'r') as seq_f:
            raw_lines = seq_f.readlines()
        raw_line = raw_lines[sample['frame_idx']]
        line = raw_line.strip().split(' ')
        trans_matrix = np.array(line[1:]).astype(np.float32)
        trans_matrix = trans_matrix.reshape(4, 4).transpose()
        # print('Loading obj transform from {}'.format(seq_path))
        return trans_matrix

    def downsample_points(self, points, depth):

        downsample_ratio_x = 1920 / 416.
        downsample_ratio_y = 1080 / 416.

        x = points[0] / downsample_ratio_x
        y = points[1] / downsample_ratio_y
        z = depth / 10.  # converting to centimeters

        downsampled_x = x / 32
        downsampled_y = y / 32
        downsampled_z = z / 15

        return downsampled_x, downsampled_y, downsampled_z

    def upsample_points(self, points, depth):

        downsample_ratio_x = 1920 / 416.
        downsample_ratio_y = 1080 / 416.

        u = points[0] * downsample_ratio_x
        v = points[1] * downsample_ratio_y
        z = depth * 10.  # converting to millimeters

        return u, v, z

    def get_cell(self, root, depth):

        downsampled_x, downsampled_y, downsampled_z = self.downsample_points(root, depth)

        u = int(downsampled_x)
        v = int(downsampled_y)
        z = int(downsampled_z)

        return (u, v, z)

    def compute_offset(self, points, cell):

        points_u, points_v, points_z = points
        points_u, points_v, points_z = self.downsample_points((points_u, points_v), points_z)
        cell_u, cell_v, cell_z = cell
        del_u = points_u - cell_u
        del_v = points_v - cell_v
        del_z = points_z - cell_z

        return del_u, del_v, del_z

    def get_box_3d_control_points(self, corners):

        # lines (0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4), (0,4), (1,5), (2,6), (3,7)

        edge_01 = (corners[0] + corners[1]) / 2.
        edge_12 = (corners[1] + corners[2]) / 2.
        edge_23 = (corners[2] + corners[3]) / 2.
        edge_30 = (corners[3] + corners[0]) / 2.
        edge_45 = (corners[4] + corners[5]) / 2.
        edge_56 = (corners[5] + corners[6]) / 2.
        edge_67 = (corners[6] + corners[7]) / 2.
        edge_74 = (corners[7] + corners[4]) / 2.
        edge_04 = (corners[0] + corners[4]) / 2.
        edge_15 = (corners[1] + corners[5]) / 2.
        edge_26 = (corners[2] + corners[6]) / 2.
        edge_37 = (corners[3] + corners[7]) / 2.

        center = np.mean(corners, axis=0)

        control_points = np.vstack((center, corners,
                                    edge_01, edge_12, edge_23, edge_30,
                                    edge_45, edge_56, edge_67, edge_74,
                                    edge_04, edge_15, edge_26, edge_37))

        return control_points

    def control_to_target(self, projected_points, points):
        # 2D location : projected_points[0, :]
        # depth : points[0, 2]
        root = projected_points[0, :]

        # get location of cell in (13, 13, 5)
        cell = self.get_cell(root, points[0, 2])

        points = projected_points[:, 0], projected_points[:, 1], points[:, 2]  # px, px, mm

        del_u, del_v, del_z = self.compute_offset(points, cell)

        return del_u, del_v, del_z, cell

    def target_to_control(self, del_u, del_v, del_z, cell):

        u, v, z = cell

        w_u = del_u + u
        w_v = del_v + v
        w_z = del_z + z

        w_u, w_v, w_z = self.upsample_points((w_u, w_v), w_z)

        ones = np.ones((21, 1), dtype=np.float32)

        points = np.vstack((w_u * 32, w_v * 32, np.ones_like(w_u)))

        y_hat = w_z * 15 * np.linalg.inv(self.camera_intrinsics).dot(points)

        return y_hat.T, points


if __name__ == '__main__':
    train = UnifiedPoseDataset(mode='test', loadit=False, name='test')
    train[0]
