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
from vis_utils import fh_utils as Frei_util

from positional_encodings import PositionalEncoding1D, PositionalEncoding2D, PositionalEncoding3D
import albumentations as album
import tqdm

from obman_util.obman import ObMan
from obman_util.visutils import visualize_2d, visualize_3d


def _assert_exist(p):
    msg = 'File does not exists: %s' % p
    assert os.path.exists(p), msg


jointsMapManoToSimple = [0,
                         13, 14, 15, 16,
                         1, 2, 3, 17,
                         4, 5, 6, 18,
                         10, 11, 12, 19,
                         7, 8, 9, 20]

jointsMapSimpleToMano = [0,
                         5, 6, 7,
                         9, 10, 11,
                         17, 18, 19,
                         13, 14, 15,
                         1, 2, 3,
                         4, 8, 12, 16, 20]

jointsMapFHADToSimple = [0,
                        1, 6, 7, 8,
                        2, 9, 10, 11,
                        3, 12, 13, 14,
                        4, 15, 16, 17,
                        5, 18, 19, 20]

VISIBLE_PARAM = 0.025
crop_threshold = 80

def _crop_imageandkeypoints(bb, image, downsample_ratio_x, downsample_ratio_y, keypoints=None):
    x_min, y_min, x_max, y_max = bb
    x_min /= downsample_ratio_x
    x_max /= downsample_ratio_x
    y_min /= downsample_ratio_y
    y_max /= downsample_ratio_y

    if keypoints is None:
        image = image[max(0, int(y_min) - crop_threshold):min(416, int(y_max) + crop_threshold),
                max(0, int(x_min) - crop_threshold):min(416, int(x_max) + crop_threshold), :]
        cropped_size = image.shape

        image = cv2.resize(image, dsize=(416, 416), interpolation=cv2.INTER_CUBIC)

        return image, cropped_size, x_min, y_min
    else:
        image = image[max(0, int(y_min) - crop_threshold):min(416, int(y_max) + crop_threshold),
                max(0, int(x_min) - crop_threshold):min(416, int(x_max) + crop_threshold), :]
        cropped_size = image.shape

        image = cv2.resize(image, dsize=(416, 416), interpolation=cv2.INTER_CUBIC)

        for kps in keypoints:
            kps[:, 0] /= downsample_ratio_x
            kps[:, 1] /= downsample_ratio_y

            kps[:, 0] = (kps[:, 0] - max(0, x_min - crop_threshold)) * (416 / cropped_size[1])
            kps[:, 1] = (kps[:, 1] - max(0, y_min - crop_threshold)) * (416 / cropped_size[0])

        # imgAnno = showHandJoints(image, handKps)  # showHandJoints_vis(img, xy_points, visible)
        # imgAnno_rgb = imgAnno[:, :, [2, 1, 0]]
        # cv2.imshow("resized crop rgb + handKps", imgAnno_rgb)
        # cv2.waitKey(0)

        return image, cropped_size, keypoints, x_min, y_min


class HO3D_v2_Dataset(Dataset):

    def __init__(self, mode='train', root='../../dataset/HO3D_V2', cfg='train', loadit=False, augment=False, extra=False, small=False):
        ###
        # initial setting
        # 640*480 image to 32*32*5 box
        # only control depth_discretization parameter(=5) in cfg.yaml
        # hand joint order : Mano
        ###
        self.coord_change_mat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)

        self.name = cfg
        self.root = root
        self.loadit = loadit
        self.mode = mode
        self.augment = augment
        self.extra = extra

        self.transform = transforms.ColorJitter(0.5, 0.5, 0.5)
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        self.IMAGE_WIDTH = 640.
        self.IMAGE_HEIGHT = 480.

        self.downsample_ratio_x = self.IMAGE_WIDTH / 416.
        self.downsample_ratio_y = self.IMAGE_HEIGHT / 416.

        #
        self.albumtransform = album.Compose([album.HorizontalFlip(p=0.5), album.VerticalFlip(p=0.5), album.Transpose(p=0.5), album.Rotate(p=0.3)],
                                  keypoint_params=album.KeypointParams(format='xy', remove_invisible=False))
        self.albumtransform_img = album.Compose([album.CoarseDropout(max_holes=3, max_height=40, max_width=40, min_holes=1, min_height=10, min_width=10, p=0.3)])

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

        # self.camera_intrinsics = np.array([[1395.749023, 0, 935.732544],
        #                                    [0, 1395.749268, 540.681030],
        #                                    [0, 0, 1]])
        self.camera_intrinsics = np.array([[617.287, 0, 314.564],
                                           [0, 617.061, 236.353],
                                           [0, 0, 1]])

        # self.depth_intrinsics = np.array([[475.065948, 0, 315.944855],
        #                                    [0,  475.065857, 245.287079],
        #                                    [0, 0, 1]])

        p_enc_3d = PositionalEncoding3D(21)
        z = torch.zeros((1, 13, 13, 5, 21))
        self.pos_encoder = p_enc_3d(z)

        self.prev_frame_idx = 0

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

                data_split = int(frames)  # * 1 / 30) + 1
                data_split_train = int(data_split * 7 / 8) + 1
                data_split_valid = data_split - data_split_train
                valid_set = np.random.choice(data_split, data_split_valid)

                for i in range(frames):
                    if i in valid_set:
                        dataset['test'][subject].append(rgb_set[i])
                    else:
                        dataset['train'][subject].append(rgb_set[i])

            #print(yaml.dump(dataset))

            modes = ['train', 'test']
            for i in range(2):
                self.samples = dict()
                self.name = modes[i]
                idx = 0
                for subject in list(dataset[modes[i]]):
                    for frame in dataset[modes[i]][subject]:
                        sample = {
                            'subject': subject,
                            'frame_idx': frame[:-4],
                        }
                        _, depth, meta = self.read_data(sample)
                        self.camera_intrinsics = meta['camMat']

                        objCorners = meta['objCorners3DRest']
                        objCornersTrans = np.matmul(objCorners, cv2.Rodrigues(meta['objRot'])[0].T) + meta['objTrans']
                        objCornersTrans = objCornersTrans.dot(self.coord_change_mat.T) * 1000.
                        objJoints3D = self.get_box_3d_control_points(objCornersTrans)

                        handJoints3D = meta['handJoints3D']
                        handJoints3D = handJoints3D.dot(self.coord_change_mat.T) * 1000.

                        objKps = project_3D_points(meta['camMat'], objJoints3D, is_OpenGL_coords=False)
                        handKps = project_3D_points(meta['camMat'], handJoints3D, is_OpenGL_coords=False)

                        handKps = handKps[jointsMapManoToSimple]
                        handJoints3D = handJoints3D[jointsMapManoToSimple]

                        ### generate visibility value ###
                        handKps_ = np.copy(handKps)
                        handJoints3D_ = np.copy(handJoints3D) / 1000.  # mm to GT unit
                        handKps_ = np.round(handKps_).astype(np.int)
                        visible = []
                        for i_vis in range(21):
                            if handKps_[i_vis][0] >= self.IMAGE_WIDTH or handKps_[i_vis][1] >= self.IMAGE_HEIGHT:
                                continue
                            d_img = depth[handKps_[i_vis][1], handKps_[i_vis][0]]
                            d_gt = handJoints3D_[i_vis][-1]
                            if np.abs(d_img - d_gt) < VISIBLE_PARAM:
                                visible.append(i_vis)

                        ### generate BoundingBox ###
                        x_min = int(np.min(handKps[:, 0]))
                        x_max = int(np.max(handKps[:, 0]))
                        y_min = int(np.min(handKps[:, 1]))
                        y_max = int(np.max(handKps[:, 1]))

                        bbox = [x_min, y_min, x_max, y_max]

                        new_sample = {
                            'subject': subject,
                            'frame_idx': frame[:-4],
                            'handJoints3D': handJoints3D,
                            'handKps': handKps,
                            'objJoints3D': objJoints3D,
                            'objKps': objKps,
                            'visible': visible,
                            'bb': bbox
                        }
                        self.samples[idx] = new_sample
                        idx += 1
                        if idx % 1000 == 0:
                            print("preprocessing idx : ", idx)

                self.clean_data()
                self.save_samples()

        else:
            self.samples = self.load_samples()
            ### test meta data has missing annotation, only acquire images in 'train' folder ###
            #self.mode = 'train'

            if small:
                sample_len = int(len(self.samples) / 8)
                self.samples = self.samples[:sample_len]

            self.sample_len = len(self.samples)


    def load_samples(self):
        with open('../cfg/HO3D_v2/{}.pkl'.format(self.name), 'rb') as f:
            samples = pickle.load(f)
            return samples

    def save_samples(self):
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
        img = self.transform(img)
        img = np.asarray(img.resize((416, 416), Image.ANTIALIAS), dtype=np.float32)
        if img.shape[-1] != 3:
            img = img[:, :, :-1]

        img = img / 255.
        # cv2.imshow("img in dataset", img)
        # cv2.waitKey(0)

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

        handKps = np.copy(sample['handKps'])
        handJoints3D = np.copy(sample['handJoints3D'])
        objKps = np.copy(sample['objKps'])
        objJoints3D = np.copy(sample['objJoints3D'])
        visible = np.copy(sample['visible'])

        # handKps = sample['handKps']
        # handJoints3D = sample['handJoints3D']
        # objKps = sample['objKps']
        # objJoints3D = sample['objJoints3D']
        # visible = sample['visible']

        ### check GT values ###
        # depth_proc = np.copy(depth)
        # depth_proc[depth > 1.0] = 0.0
        # # depthAnno = showHandJoints(depth_proc, handKps_)
        # imgAnno = showHandJoints_vis(img, handKps_, vis=visible)
        # # imgAnno = showHandJoints(img, handKps)
        # # imgAnno = showObjJoints(imgAnno, objKps, lineThickness=2)
        # # rgb = img[:, :, [0, 1, 2]]
        # # cv2.imshow("rgb", rgb)
        # anno = imgAnno[:, :, [0, 1, 2]]
        # cv2.imshow("anno", anno)
        # # cv2.imshow("depthAnno", depthAnno)
        # cv2.waitKey(0)


        ### preprocess extrapolated keypoint GT ###
        image = None
        flag_nonzero_extra = False

        if self.loadit and self.extra:
            prev_handKps = dict()
            prev_handJoints3D = dict()
            prev_vis = dict()
            if idx < 2:
                for i in range(2):
                    prev_handKps[i] = np.zeros((21, 2), dtype=np.float32)
                    prev_handJoints3D[i] = np.zeros((21, 3), dtype=np.float32)
                    prev_vis[i] = np.zeros(21, dtype=np.float32)
            else:
                prev_sample = self.samples[idx - 2]
                prev_frame_idx = int(prev_sample['frame_idx'])
                if int(frame_idx - prev_frame_idx) != 2:
                    for i in range(2):
                        prev_handKps[i] = np.zeros((21, 2), dtype=np.float32)
                        prev_handJoints3D[i] = np.zeros((21, 3), dtype=np.float32)
                        prev_vis[i] = np.zeros(21, dtype=np.float32)
                else:
                    flag_nonzero_extra = True

                    flag_augment = False
                    if idx > 20:
                        old_sample = self.samples[idx - 20]
                        old_frame_idx = int(old_sample['frame_idx'])
                        if int(frame_idx - old_frame_idx) != 20:
                            flag_augment = random.choices(population=[False, True], weights=[0.9, 0.1], k=1)[0]

                    for i in range(2):
                        if flag_augment:
                            prev_sample = old_sample
                        else:
                            prev_sample = self.samples[idx - (i + 1)]

                        prev_handKps[i] = np.copy(prev_sample['handKps'])
                        prev_handJoints3D[i] = np.copy(prev_sample['handJoints3D'])

            root_diff = prev_handKps[0][0, :] - prev_handKps[1][0, :]

            ### relative extrapolation ###
            prev_handKps[1] += root_diff

            mixed_handKps = 2 * prev_handKps[0] - prev_handKps[1]
            mixed_handJoints3D = 2 * prev_handJoints3D[0] - prev_handJoints3D[1]

            # if mixed_handKps[0, 0] < 0 or mixed_handKps[0, 1] < 0:
            #     print("ss")
            #     print("sdf")

            ### absolute extrapolation (original) ###
            """
            # dist = np.sqrt(root_diff[0]*root_diff[0] + root_diff[1]*root_diff[1])
            # if dist < 10.:
            #     mixed_handKps = 2 * prev_handKps[0] - prev_handKps[1]
            #     mixed_handJoints3D = 2 * prev_handJoints3D[0] - prev_handJoints3D[1]
            # else:
            #     mixed_handKps = np.copy(prev_handKps[0])
            #     mixed_handJoints3D = np.copy(prev_handJoints3D[0])

            # visible_index = np.multiply(prev_vis[0], prev_vis[1]).astype(int)
            #
            # mixed_handKps[visible_index, :] = extra_handKps[visible_index, :]
            # mixed_handJoints3D[visible_index, :] = extra_handJoints3D[visible_index, :]
            # mixed_handKps[:, 0] /= self.downsample_ratio_x
            # mixed_handKps[:, 1] /= self.downsample_ratio_y
            """

        ### Augmentation with GTs ###
        if self.loadit:
            image = self.get_image(sample)
            bb = np.copy(sample['bb'])

            if self.extra and flag_nonzero_extra:
                keypoints = [handKps, objKps, mixed_handKps]
            else:
                keypoints = [handKps, objKps]

            ### image is already resized ###
            image, cropped_size, keypoints, x_min, y_min = _crop_imageandkeypoints(bb, image, self.downsample_ratio_x, self.downsample_ratio_y, keypoints=keypoints)

            crop_param = [x_min, y_min, cropped_size]

            if len(keypoints) is 3 and flag_nonzero_extra:
                [handKps, objKps, mixed_handKps] = keypoints
            else:
                [handKps, objKps] = keypoints

            # imgAnno = showHandJoints(image, handKps)  # showHandJoints_vis(img, xy_points, visible)
            # imgAnno_rgb = imgAnno[:, :, [2, 1, 0]]
            # cv2.imshow("rgb pred", imgAnno_rgb)
            # cv2.waitKey(1)
            # if mixed_handKps[0, 0] < 0 or mixed_handKps[0, 1] < 0:
            #     print("ss")
            #     print("sdf")

            if self.augment:
                if self.extra and flag_nonzero_extra:
                    noise = np.random.normal(0, 1., mixed_handKps.shape)
                    mixed_handKps = mixed_handKps + noise
                    Kps = np.concatenate((handKps, objKps, mixed_handKps), axis=0)

                    transformed = self.albumtransform(image=image, keypoints=Kps)

                    image = transformed['image']
                    Kps = np.array(transformed['keypoints'], dtype='float')
                    handKps = Kps[:21, :]
                    objKps = Kps[21:42, :]
                    mixed_handKps = Kps[42:, :]

                else:
                    Kps = np.concatenate((handKps, objKps), axis=0)

                    transformed = self.albumtransform(image=image, keypoints=Kps)

                    image = transformed['image']
                    Kps = np.array(transformed['keypoints'], dtype='float')
                    handKps = Kps[:21, :]
                    objKps = Kps[21:, :]



                ### random crop out image region ###
                transformed = self.albumtransform_img(image=image)
                image = transformed['image']

            image = np.squeeze(np.transpose(image, (2, 0, 1)))
            image = torch.from_numpy(image)
            if int(image.shape[0]) != 3:
                print("image shape wrong")
                image = image[:-1, :, :]
            image = self.normalize(image)

        ### object pose ###
        # get offset w.r.t top/left corner of the cell
        del_u, del_v, del_z, cell = self.control_to_target(objKps, objJoints3D, self.loadit)

        # object pose tensor
        true_object_pose = torch.zeros(21, 3, 5, 13, 13, dtype=torch.float32)
        u, v, z = cell
        pose = np.vstack((del_u, del_v, del_z)).T
        # try:
        true_object_pose[:, :, z, v, u] = torch.from_numpy(pose)
        # except Exception as e:
        #     print(e)

        true_object_pose = true_object_pose.view(-1, 5, 13, 13)     # (63, 5, 13, 13)

        # object mask
        object_mask = torch.zeros(5, 13, 13, dtype=torch.float32)
        object_mask[z, v, u] = 1

        ### hand pose ###
        del_u, del_v, del_z, cell = self.control_to_target(handKps, handJoints3D, self.loadit)  # handKps : [215.8, 182.1] , ...   /  handJoints3D : [~, ~, 462.2] , ...
        # hand pose tensor
        true_hand_pose = torch.zeros(21, 3, 5, 13, 13, dtype=torch.float32)
        u, v, z = cell
        pose = np.vstack((del_u, del_v, del_z)).T

        # try:
        true_hand_pose[:, :, z, v, u] = torch.from_numpy(pose)
        # except Exception as e:
        #     print(e)

        true_hand_pose = true_hand_pose.view(-1, 5, 13, 13)
        # hand mask
        hand_mask = torch.zeros(5, 13, 13, dtype=torch.float32)
        hand_mask[z, v, u] = 1

        ### hand visibility per joint ###
        vis = np.zeros(21, dtype=np.float32)
        if len(visible) != 0:
            vis[np.array(visible)] = 1

        param_vis = torch.zeros(21, 5, 13, 13, dtype=torch.float32)
        param_vis[:, z, v, u] = torch.from_numpy(vis)
        param_vis = param_vis.view(-1, 5, 13, 13)

        ### extrapolated hand pose ###
        extra_hand_pose = torch.zeros(4, 21, dtype=torch.float32)
        if self.loadit and self.extra:
            del_u, del_v, del_z, cell = self.control_to_target(mixed_handKps, mixed_handJoints3D, True)

            # hand pose tensor
            # index + del, with positional encoding
            del_u = torch.unsqueeze(torch.from_numpy(del_u), 0).type(torch.float32)
            del_v = torch.unsqueeze(torch.from_numpy(del_v), 0).type(torch.float32)
            del_z = torch.unsqueeze(torch.from_numpy(del_z), 0).type(torch.float32)

            # try:
            enc_cell = self.pos_encoder[:, cell[0], cell[1], cell[2], :].type(torch.float32)
            # except Exception as e:
            #     print(e)

            extra_hand_pose = torch.cat((enc_cell, del_u, del_v, del_z), 0)  # tensor (4, 21)

        return image, true_hand_pose, hand_mask, true_object_pose, object_mask, param_vis, vis, extra_hand_pose, crop_param


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

        x = points[0]   # / self.downsample_ratio_x    ## already downsampled
        y = points[1]   # / self.downsample_ratio_y
        z = depth / 10.  # converting to centimeters

        downsampled_x = x / 32
        downsampled_y = y / 32
        downsampled_z = z / 25

        return downsampled_x, downsampled_y, downsampled_z

    def upsample_points(self, points, depth):
        u = points[0] * self.downsample_ratio_x
        v = points[1] * self.downsample_ratio_y
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

    def control_to_target(self, projected_points, points, loadit):
        # 2D location : projected_points[0, :]
        # depth : points[0, 2]
        root = projected_points[0, :]

        # get location of cell in (13, 13, 5)
        cell = self.get_cell(root, points[0, 2])
        if loadit:
            u, v, z = cell
            u = np.clip(u, 0, 12)
            v = np.clip(v, 0, 12)
            z = np.clip(z, 0, 4)
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
        y_hat = w_z * 25 * np.linalg.inv(self.camera_intrinsics).dot(points)

        return y_hat.T, points

class FHAD_Dataset(Dataset):

    def __init__(self, mode='train', root='../../dataset/First_Person_Action_Benchmark-selected', loadit=False, cfg=None, extra=False, augment=False, small=False):
        ###
        # initial setting
        # 1920*1080 image to 32*32*5 box
        # only control depth_discretization parameter(=5) in cfg.yaml
        ###
        self.IMAGE_WIDTH = 1920.
        self.IMAGE_HEIGHT = 1080.

        self.name = cfg
        self.root = root
        self.loadit = loadit
        self.mode = mode
        self.extra = extra
        self.augment = augment

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


        self.downsample_ratio_x = self.IMAGE_WIDTH / 416.
        self.downsample_ratio_y = self.IMAGE_HEIGHT / 416.

        self.albumtransform = album.Compose([album.HorizontalFlip(p=0.5), album.VerticalFlip(p=0.5),
                                             album.Transpose(p=0.5), album.Rotate(p=0.3)],
                                            keypoint_params=album.KeypointParams(format='xy', remove_invisible=False))
        self.albumtransform_img = album.Compose([album.CoarseDropout(max_holes=3, max_height=40, max_width=40,
                                                                     min_holes=1, min_height=10, min_width=10, p=0.3)])

        p_enc_3d = PositionalEncoding3D(21)
        z = torch.zeros((1, 13, 13, 5, 21))
        self.pos_encoder = p_enc_3d(z)

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

                #random.shuffle(actions)

                for action in actions:
                    dataset['train'][subject][action] = dict()
                    dataset['test'][subject][action] = dict()

                    pose_sequences = set(
                        os.listdir(os.path.join(root, 'Object_6D_pose_annotation_v1', subject, action)))
                    video_sequences = set(os.listdir(os.path.join(root, 'Video_files', subject, action)))
                    sequences = list(pose_sequences.intersection(video_sequences))

                    data_split = int(len(sequences)) + 1 # * 3 / 4) + 1

                    #random.shuffle(sequences)

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


            #print(yaml.dump(dataset))
            idx = 0
            modes = ['train', 'test']
            skeleton_root = os.path.join(self.root, 'Hand_pose_annotation_v1')
            object_pose_root = os.path.join(self.root, 'Object_6D_pose_annotation_v1')

            for i in range(2):
                self.samples = dict()
                self.name = modes[i]

                for subject in list(dataset[modes[i]]):
                    for action in list(dataset[modes[i]][subject]):
                        for sequence in list(dataset[modes[i]][subject][action]):
                            for frame in dataset[modes[i]][subject][action][sequence]:
                                sample = {
                                    'subject': subject,
                                    'action_name': action,
                                    'seq_idx': str(sequence),
                                    'frame_idx': frame,
                                    'object': action_to_object[action]
                                }

                                # Hand Properties
                                handJoints3D = self.get_skeleton(sample, skeleton_root)[jointsMapFHADToSimple]
                                homogeneous_skeleton = np.concatenate(
                                    [handJoints3D, np.ones([handJoints3D.shape[0], 1])], 1)
                                handJoints3D = self.camera_pose.dot(homogeneous_skeleton.T).T[:, :3].astype(
                                    np.float32)  # mm unit
                                homogeneous_handKps = np.array(self.camera_intrinsics).dot(handJoints3D.T).T
                                handKps = (homogeneous_handKps / homogeneous_handKps[:, 2:])[:, :2]

                                # Object Properties
                                object_pose = self.get_object_pose(sample, object_pose_root)
                                corners = self.objects[sample['object']]['corners'] * 1000.
                                homogeneous_corners = np.concatenate([corners, np.ones([corners.shape[0], 1])], axis=1)
                                corners = object_pose.dot(homogeneous_corners.T).T
                                corners = self.camera_pose.dot(corners.T).T[:, :3]
                                objJoints3D = self.get_box_3d_control_points(corners)
                                homogeneous_control_points = np.array(self.camera_intrinsics).dot(objJoints3D.T).T
                                objKps = (homogeneous_control_points / homogeneous_control_points[:, 2:])[:, :2]

                                ### generate BoundingBox ###
                                x_min = int(np.min(handKps[:, 0]))
                                x_max = int(np.max(handKps[:, 0]))
                                y_min = int(np.min(handKps[:, 1]))
                                y_max = int(np.max(handKps[:, 1]))

                                bbox = [x_min, y_min, x_max, y_max]

                                new_sample = {
                                    'subject': subject,
                                    'action_name': action,
                                    'seq_idx': str(sequence),
                                    'frame_idx': frame,
                                    'handJoints3D': handJoints3D,
                                    'handKps': handKps,
                                    'objJoints3D': objJoints3D,
                                    'objKps': objKps,
                                    'bb': bbox
                                }

                                self.samples[idx] = new_sample
                                idx += 1
                                if idx % 1000 == 0:
                                    print("preprocessing idx : ", idx)
                self.clean_data()
                self.save_samples(modes[i])

        else:
            self.samples = self.load_samples(mode)

            if small:
                sample_len = int(len(self.samples) / 8)
                self.samples = self.samples[:sample_len]


    def load_samples(self, mode):
        with open('../cfg/FHAD/{}.pkl'.format(self.name), 'rb') as f:
            samples = pickle.load(f)
            return samples

    def save_samples(self, mode):
        with open('../cfg/FHAD/{}.pkl'.format(self.name), 'wb') as f:
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

        # cv2.imshow("rgb in dataset", img)
        # cv2.waitKey(0)

        #img = np.transpose(img, (2, 0, 1))
        return img

    def get_depth(self, sample):

        img = self.fetch_depth(sample)

        img = np.asarray(img.resize((416, 416), Image.ANTIALIAS), dtype=np.float32)
        #img = np.flip(img, axis=1)
        img = img / 255.

        # cv2.imshow("depth in dataset", img)
        # cv2.waitKey(1)

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
        # object_category = {
        #     'juice': 0,
        #     'liquid_soap': 1,
        #     'milk': 2,
        #     'salt': 3
        # }
        #
        # action_category = {
        #     'open_milk': 0,
        #     'close_milk': 1,
        #     'pour_milk': 2,
        #     'open_juice_bottle': 0,
        #     'close_juice_bottle': 1,
        #     'pour_juice_bottle': 2,
        #     'open_liquid_soap': 0,
        #     'close_liquid_soap': 1,
        #     'pour_liquid_soap': 2,
        #     'put_salt': 3
        # }
        sample = self.samples[idx]
        frame_idx = int(sample['frame_idx'])

        handKps = np.copy(sample['handKps'])
        handJoints3D = np.copy(sample['handJoints3D'])
        objKps = np.copy(sample['objKps'])
        objJoints3D = np.copy(sample['objJoints3D'])

        # ### rescale keypoints first for augmentation with image correspondence ###
        # handKps[:, 0] = handKps[:, 0] / self.downsample_ratio_x
        # handKps[:, 1] = handKps[:, 1] / self.downsample_ratio_y
        # objKps[:, 0] = objKps[:, 0] / self.downsample_ratio_x
        # objKps[:, 1] = objKps[:, 1] / self.downsample_ratio_y

        ### preprocess extrapolated keypoint GT ... without visibility for FHAD ###
        flag_nonzero_extra = False
        if self.loadit and self.extra:
            prev_handKps = dict()
            prev_handJoints3D = dict()
            prev_vis = dict()

            if idx < 2:
                for i in range(2):
                    prev_handKps[i] = np.zeros((21, 2), dtype=np.float32)
                    prev_handJoints3D[i] = np.zeros((21, 3), dtype=np.float32)
                    prev_vis[i] = np.zeros(21, dtype=np.float32)

            else:
                prev_sample = self.samples[idx - 2]
                prev_frame_idx = int(prev_sample['frame_idx'])
                if int(frame_idx - prev_frame_idx) != 2:
                    for i in range(2):
                        prev_handKps[i] = np.zeros((21, 2), dtype=np.float32)
                        prev_handJoints3D[i] = np.zeros((21, 3), dtype=np.float32)
                        prev_vis[i] = np.zeros(21, dtype=np.float32)
                else:
                    flag_nonzero_extra = True

                    flag_old_augment = False
                    if idx > 20:
                        old_sample = self.samples[idx - 20]
                        old_frame_idx = int(old_sample['frame_idx'])
                        if int(frame_idx - old_frame_idx) != 20:
                            flag_old_augment = random.choices(population=[False, True], weights=[0.9, 0.1], k=1)[0]

                    for i in range(2):
                        if flag_old_augment:
                            prev_sample = old_sample
                        else:
                            prev_sample = self.samples[idx - (i + 1)]

                        prev_handKps[i] =  np.copy(prev_sample['handKps'])
                        prev_handJoints3D[i] = np.copy(prev_sample['handJoints3D'])

            root_diff = prev_handKps[0][0, :] - prev_handKps[1][0, :]

            ### relative extrapolation ###
            prev_handKps[1] += root_diff

            mixed_handKps = 2 * prev_handKps[0] - prev_handKps[1]
            mixed_handJoints3D = 2 * prev_handJoints3D[0] - prev_handJoints3D[1]
            """
            root = prev_handKps[0][0, :] - prev_handKps[1][0, :]
            dist = np.sqrt(root[0] * root[0] + root[1] * root[1])

            if dist < 10.:
                mixed_handKps = 2 * prev_handKps[0] - prev_handKps[1]
                mixed_handJoints3D = 2 * prev_handJoints3D[0] - prev_handJoints3D[1]
            else:
                mixed_handKps = np.copy(prev_handKps[0])
                mixed_handJoints3D = np.copy(prev_handJoints3D[0])

            mixed_handKps[:, 0] = mixed_handKps[:, 0] / self.downsample_ratio_x
            mixed_handKps[:, 1] = mixed_handKps[:, 1] / self.downsample_ratio_y
            """

        ### Augmentation with GTs ###
        if self.loadit:
            image = self.get_image(sample)
            bb = np.copy(sample['bb'])

            if self.extra:
                keypoints = [handKps, objKps, mixed_handKps]
            else:
                keypoints = [handKps, objKps]

            ### image is already resized ###
            image, cropped_size, keypoints, x_min, y_min = _crop_imageandkeypoints(bb, image, self.downsample_ratio_x, self.downsample_ratio_y, keypoints=keypoints)

            crop_param = [x_min, y_min, cropped_size]

            if len(keypoints) is 3:
                [handKps, objKps, mixed_handKps] = keypoints
            else:
                [handKps, objKps] = keypoints

            # imgAnno = showHandJoints(image, handKps)  # showHandJoints_vis(img, xy_points, visible)
            # imgAnno_rgb = imgAnno[:, :, [2, 1, 0]]
            # cv2.imshow("rgb pred", imgAnno_rgb)
            # cv2.waitKey(1)

            if self.augment:
                if self.extra and flag_nonzero_extra:
                    noise = np.random.normal(0, 1., mixed_handKps.shape)
                    mixed_handKps = mixed_handKps + noise
                    Kps = np.concatenate((handKps, objKps, mixed_handKps), axis=0)

                    transformed = self.albumtransform(image=image, keypoints=Kps)

                    image = transformed['image']
                    Kps = np.array(transformed['keypoints'], dtype='float')
                    handKps = Kps[:21, :]
                    objKps = Kps[21:42, :]
                    mixed_handKps = Kps[42:, :]

                else:
                    Kps = np.concatenate((handKps, objKps), axis=0)

                    transformed = self.albumtransform(image=image, keypoints=Kps)

                    image = transformed['image']
                    Kps = np.array(transformed['keypoints'], dtype='float')
                    handKps = Kps[:21, :]
                    objKps = Kps[21:, :]

                ### random crop out image region ###
                transformed = self.albumtransform_img(image=image)
                image = transformed['image']

            image = np.squeeze(np.transpose(image, (2, 0, 1)))
            image = torch.from_numpy(image)
            if int(image.shape[0]) != 3:
                print("image shape wrong")
                image = image[:-1, :, :]
            image = self.normalize(image)

        # hand pose tensor
        del_u, del_v, del_z, cell = self.control_to_target(handKps, handJoints3D, self.loadit)

        true_hand_pose = torch.zeros(21, 3, 5, 13, 13, dtype=torch.float32)
        u, v, z = cell
        pose = np.vstack((del_u, del_v, del_z)).T
        true_hand_pose[:, :, z, v, u] = torch.from_numpy(pose)
        true_hand_pose = true_hand_pose.view(-1, 5, 13, 13)

        # hand mask
        hand_mask = torch.zeros(5, 13, 13, dtype=torch.float32)
        hand_mask[z, v, u] = 1

        # object pose tensor
        del_u, del_v, del_z, cell = self.control_to_target(objKps, objJoints3D, self.loadit)

        true_object_pose = torch.zeros(21, 3, 5, 13, 13, dtype=torch.float32)
        u, v, z = cell
        pose = np.vstack((del_u, del_v, del_z)).T
        true_object_pose[:, :, z, v, u] = torch.from_numpy(pose)
        true_object_pose = true_object_pose.view(-1, 5, 13, 13)  # (63, 5, 13, 13)

        # object mask
        object_mask = torch.zeros(5, 13, 13, dtype=torch.float32)
        object_mask[z, v, u] = 1

        # # hand action tensor
        # true_hand_prob = torch.zeros(5, 13, 13, dtype=torch.long)
        # true_hand_prob[z, v, u] = action_category[sample['action_name']]
        # # object class tensor
        # true_object_prob = torch.zeros(5, 13, 13, dtype=torch.long)
        # true_object_prob[z, v, u] = object_category[sample['object']]

        ### extrapolated hand pose ###
        extra_hand_pose = torch.zeros(4, 21, dtype=torch.float32)
        if self.loadit and self.extra:
            del_u, del_v, del_z, cell = self.control_to_target(mixed_handKps, mixed_handJoints3D, True)

            # hand pose tensor
            # index + del, with positional encoding
            del_u = torch.unsqueeze(torch.from_numpy(del_u), 0).type(torch.float32)
            del_v = torch.unsqueeze(torch.from_numpy(del_v), 0).type(torch.float32)
            del_z = torch.unsqueeze(torch.from_numpy(del_z), 0).type(torch.float32)

            # try:
            enc_cell = self.pos_encoder[:, cell[0], cell[1], cell[2], :].type(torch.float32)
            # except Exception as e:
            #     print(e)

            extra_hand_pose = torch.cat((enc_cell, del_u, del_v, del_z), 0)  # tensor (4, 21)

        return image, true_hand_pose, hand_mask, true_object_pose, object_mask, extra_hand_pose, crop_param

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

        # downsample_ratio_x = 1920 / 416.
        # downsample_ratio_y = 1080 / 416.

        x = points[0] #/ downsample_ratio_x
        y = points[1] #/ downsample_ratio_y
        z = depth / 10.  # converting to centimeters

        downsampled_x = x / 32
        downsampled_y = y / 32
        downsampled_z = z / 25

        return downsampled_x, downsampled_y, downsampled_z

    def upsample_points(self, points, depth):

        u = points[0] * self.downsample_ratio_x
        v = points[1] * self.downsample_ratio_y
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

    def control_to_target(self, projected_points, points, loadit):
        # 2D location : projected_points[0, :]
        # depth : points[0, 2]
        root = projected_points[0, :]

        # get location of cell in (13, 13, 5)
        cell = self.get_cell(root, points[0, 2])
        if loadit:
            u, v, z = cell
            u = np.clip(u, 0, 12)
            v = np.clip(v, 0, 12)
            z = np.clip(z, 0, 4)
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

        ones = np.ones((21, 1), dtype=np.float32)

        points = np.vstack((w_u * 32, w_v * 32, np.ones_like(w_u)))

        y_hat = w_z * 25 * np.linalg.inv(self.camera_intrinsics).dot(points)

        return y_hat.T, points

class FreiHAND_Dataset(Dataset):

    def __init__(self, mode='train', root='../../dataset/FreiHAND', loadit=False, augment=False, small=False):
        ###
        # initial setting
        # 224*224 image to 32*32*5 box
        # only control depth_discretization parameter(=5) in cfg.yaml
        # hand joint order : Simple
        ###

        self.IMAGE_WIDTH = 224.
        self.IMAGE_HEIGHT = 224.

        self.coord_change_mat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)

        self.cfg = 'training'
        self.root = root
        self.loadit = loadit
        self.mode = mode
        self.augment = augment

        self.transform = transforms.ColorJitter(0.5, 0.5, 0.5)
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])



        self.downsample_ratio_x = self.IMAGE_WIDTH / 416.
        self.downsample_ratio_y = self.IMAGE_HEIGHT / 416.

        self.albumtransform = album.Compose([album.HorizontalFlip(p=0.5), album.VerticalFlip(p=0.5),
                                   album.Transpose(p=0.5)],
                                  keypoint_params=album.KeypointParams(format='xy', remove_invisible=False))
        self.albumtransform_img = album.Compose([album.CoarseDropout(max_holes=3, max_height=40, max_width=40, min_holes=1, min_height=10, min_width=10, p=0.3)])

        self.camera_pose = np.array(
            [[1., 0., 0., 0.],
             [0., 1., 0., 0.],
             [0., 0., 1., 0.],
             [0, 0, 0, 1]])

        p_enc_3d = PositionalEncoding3D(21)
        z = torch.zeros((1, 13, 13, 5, 21))
        self.pos_encoder = p_enc_3d(z)

        self.prev_frame_idx = 0

        if not loadit:
            # assumed paths to data containers
            k_path = os.path.join(self.root, 'training_K.json')
            xyz_path = os.path.join(self.root, 'training_xyz.json')

            K_list = Frei_util.json_load(k_path)
            xyz_list = Frei_util.json_load(xyz_path)
            # should have all the same length
            assert len(K_list) == len(xyz_list), 'Size mismatch.'

            db_size = len(K_list)   # 32560 for training

            dataset = dict()
            dataset['train'] = list()
            dataset['test'] = list()

            data_split = int(db_size)  # * 1 / 30) + 1
            data_split_train = int(data_split * 9 / 10) + 1
            data_split_valid = data_split - data_split_train
            valid_set = np.random.choice(data_split, data_split_valid)

            samples_train = dict()
            samples_test = dict()
            i_train = 0
            i_test = 0

            for set in range(3):
                for idx in range(db_size):
                    # augmented images starts 32560.jpg ~
                    file_idx = int(idx + 32560 * (set+1))
                    img_path = os.path.join(self.root, self.cfg, 'rgb', '%08d.jpg' % file_idx)
                    _assert_exist(img_path)

                    K = np.array(K_list[idx])
                    xyz = np.array(xyz_list[idx])
                    uv = Frei_util.projectPoints(xyz, K)

                    sample = {
                        'K': K,
                        'xyz': xyz,
                        'uv': uv,
                        'img_path': img_path,
                    }

                    if idx in valid_set:
                        samples_test[i_test] = sample
                        i_test += 1
                    else:
                        samples_train[i_train] = sample
                        i_train += 1

                    if i_train % 1000 == 0:
                        print("i : ", i_train)

            self.samples = samples_train
            self.clean_data()
            self.save_samples(name='train')

            self.samples = samples_test
            self.clean_data()
            self.save_samples(name='test')

        else:
            self.samples = self.load_samples(self.mode)
            self.sample_len = len(self.samples)


    def load_samples(self, name):
        with open('../cfg/FreiHand/{}.pkl'.format(name), 'rb') as f:
            samples = pickle.load(f)
            return samples

    def save_samples(self, name):
        with open('../cfg/FreiHand/{}.pkl'.format(name), 'wb') as f:
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
        img = self.transform(img)
        img = np.asarray(img.resize((416, 416), Image.ANTIALIAS), dtype=np.float32)
        if img.shape[-1] != 3:
            img = img[:, :, :-1]

        img = img / 255.
        # cv2.imshow("img in dataset", img)
        # cv2.waitKey(0)

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
        img_path = sample['img_path']
        img = Image.open(img_path)
        return img


    def preprocess(self, idx):

        sample = self.samples[idx]

        K = sample['K']
        handJoints3D = np.copy(sample['xyz']) * 1000.
        handKps = np.copy(sample['uv'])

        ### rescale keypoints first for extrapolation & augmentation ###
        handKps[:, 0] = handKps[:, 0] / self.downsample_ratio_x
        handKps[:, 1] = handKps[:, 1] / self.downsample_ratio_y

        ### Augmentation with GTs ###
        image = self.get_image(sample)
        # already processed color jitter, normalize to image
        if self.loadit and self.augment:
            Kps = handKps
            transformed = self.albumtransform(image=image, keypoints=Kps)

            image = transformed['image']
            handKps = np.array(transformed['keypoints'], dtype='float')

            ### random crop out image region ###
            transformed = self.albumtransform_img(image=image)
            image = transformed['image']

        image = np.squeeze(np.transpose(image, (2, 0, 1)))
        image = torch.from_numpy(image)
        if int(image.shape[0]) != 3:
            print("image shape wrong")
            image = image[:-1, :, :]
        image = self.normalize(image)

        #### rescale 3D keypoints for grid ###
        handJoints3D[:, 0] = handJoints3D[:, 0] / self.downsample_ratio_x
        handJoints3D[:, 1] = handJoints3D[:, 1] / self.downsample_ratio_y

        ### hand pose ###
        del_u, del_v, del_z, cell = self.control_to_target(handKps, handJoints3D, self.loadit)  # handKps : [215.8, 182.1] , ...   /  handJoints3D : [~, ~, 462.2] , ...
        # hand pose tensor
        true_hand_pose = torch.zeros(21, 3, 5, 13, 13, dtype=torch.float32)
        u, v, z = cell
        pose = np.vstack((del_u, del_v, del_z)).T

        # try:
        true_hand_pose[:, :, z, v, u] = torch.from_numpy(pose)
        # except Exception as e:
        #     print(e)

        true_hand_pose = true_hand_pose.view(-1, 5, 13, 13)
        # hand mask
        hand_mask = torch.zeros(5, 13, 13, dtype=torch.float32)
        hand_mask[z, v, u] = 1

        return image, true_hand_pose, hand_mask, K #, true_object_pose, object_mask, param_vis, vis, extra_hand_pose


    def downsample_points(self, points, depth):

        x = points[0]   # / self.downsample_ratio_x    ## already downsampled
        y = points[1]   # / self.downsample_ratio_y
        z = depth / 10.  # converting to centimeters

        downsampled_x = x / 32
        downsampled_y = y / 32
        downsampled_z = z / 25

        return downsampled_x, downsampled_y, downsampled_z

    def upsample_points(self, points, depth):
        u = points[0] * self.downsample_ratio_x
        v = points[1] * self.downsample_ratio_y
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

    def control_to_target(self, projected_points, points, loadit):
        # 2D location : projected_points[0, :]
        # depth : points[0, 2]
        root = projected_points[0, :]

        # get location of cell in (13, 13, 5)
        cell = self.get_cell(root, points[0, 2])
        if loadit:
            u, v, z = cell
            u = np.clip(u, 0, 12)
            v = np.clip(v, 0, 12)
            z = np.clip(z, 0, 4)
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
        y_hat = w_z * 25 * np.linalg.inv(self.camera_intrinsics).dot(points)

        return y_hat.T, points

    def target_to_control_wK(self, del_u, del_v, del_z, cell, K):

        u, v, z = cell

        w_u = del_u + u
        w_v = del_v + v
        w_z = del_z + z

        w_u, w_v, w_z = self.upsample_points((w_u, w_v), w_z)
        points = np.vstack((w_u * 32, w_v * 32, np.ones_like(w_u)))
        y_hat = w_z * 25 * np.linalg.inv(K).dot(points)

        return y_hat.T, points

class Obman_Dataset(Dataset):

    def __init__(self, mode='train', root='../../dataset/obman', loadit=False, augment=False, ratio=1.0):
        ###
        # initial setting
        # 224*224 image to 32*32*5 box
        # only control depth_discretization parameter(=5) in cfg.yaml
        # hand joint order : Simple
        ###

        self.IMAGE_WIDTH = 256.
        self.IMAGE_HEIGHT = 256.

        # self.coord_change_mat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)

        self.cfg = 'training'
        self.root = root
        self.loadit = loadit
        self.mode = mode
        self.augment = augment
        self.ratio = ratio

        self.transform = transforms.ColorJitter(0.5, 0.5, 0.5)
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        self.downsample_ratio_x = self.IMAGE_WIDTH / 416.
        self.downsample_ratio_y = self.IMAGE_HEIGHT / 416.

        self.albumtransform = album.Compose([album.HorizontalFlip(p=0.5), album.VerticalFlip(p=0.5),
                                   album.Transpose(p=0.5)],
                                  keypoint_params=album.KeypointParams(format='xy', remove_invisible=False))
        self.albumtransform_img = album.Compose([album.CoarseDropout(max_holes=3, max_height=40, max_width=40, min_holes=1, min_height=10, min_width=10, p=0.3)])

        self.camera_pose = np.array(
            [[1., 0., 0., 0.],
             [0., 1., 0., 0.],
             [0., 0., 1., 0.],
             [0, 0, 0, 1]])

        p_enc_3d = PositionalEncoding3D(21)
        z = torch.zeros((1, 13, 13, 5, 21))
        self.pos_encoder = p_enc_3d(z)

        ### need to execute clean.py to make cache and execute train.py ###
        self.obman_dataset_all = ObMan(
            '../../dataset/obman',
            '../../dataset/obman/shapenetcore_v2',
            split='train',
            use_cache=True,
            root_palm=False,
            mini_factor=0.5,
            mode='all')

        self.obman_dataset_hand = ObMan(
            '../../dataset/obman',
            '../../dataset/obman/shapenetcore_v2',
            split='train',
            use_cache=True,
            root_palm=False,
            mini_factor=0.5,
            mode='hand')

        if not loadit:
            dataset = dict()
            dataset['train'] = list()
            dataset['test'] = list()

            samples_train = dict()
            samples_test = dict()
            i_train = 0
            i_test = 0

            for img_idx in range(1, self.obman_dataset_all.dataset_len):
                #img = obman_dataset_all.get_image(img_idx)
                depth = self.obman_dataset_hand.get_depth(img_idx)

                # hand_verts3d = obman_dataset_all.get_verts3d(img_idx)
                # hand_faces = obman_dataset_all.get_faces3d(img_idx)
                hand_joints2d = self.obman_dataset_all.get_joints2d(img_idx)     # (21, 2)
                joint_d = np.zeros((21, 1), dtype=float)

                d_max = np.max(depth)
                for i in range(21):
                    x = int(hand_joints2d[i, 0])
                    y = int(hand_joints2d[i, 1])

                    x = np.clip(x, 0, 255)
                    y = np.clip(y, 0, 255)

                    d_value = depth[y, x]

                    if d_value == d_max:
                        d_value = depth[y, x+1]
                    if d_value == d_max:
                        d_value = depth[y, x-1]
                    if d_value == d_max:
                        d_value = depth[y+1, x]
                    if d_value == d_max:
                        d_value = depth[y-1, x]

                    joint_d[i, 0] = d_value

                hand_joints3d = np.append(hand_joints2d, joint_d, axis=1)

                # visualize_2d(
                #     img,
                #     hand_joints=hand_joints2d)
                # visualize_2d(
                #     depth,
                #     hand_joints=hand_joints2d)

                sample = {
                    'img_idx': img_idx,
                    'xyz': hand_joints3d,
                }

                samples_train[i_train] = sample
                i_train += 1

                if i_train % 1000 == 0:
                    print("i : ", i_train)

            """
            for set in range(3):
                for idx in range(db_size):
                    # augmented images starts 32560.jpg ~
                    file_idx = int(idx + 32560 * (set+1))
                    img_path = os.path.join(self.root, self.cfg, 'rgb', '%08d.jpg' % file_idx)
                    _assert_exist(img_path)

                    K = np.array(K_list[idx])
                    xyz = np.array(xyz_list[idx])
                    uv = Frei_util.projectPoints(xyz, K)

                    sample = {
                        'K': K,
                        'xyz': xyz,
                        'uv': uv,
                        'img_path': img_path,
                    }

                    if idx in valid_set:
                        samples_test[i_test] = sample
                        i_test += 1
                    else:
                        samples_train[i_train] = sample
                        i_train += 1
            """
            self.samples = samples_train
            self.clean_data()
            self.save_samples(name='train')

            # self.samples = samples_test
            # self.clean_data()
            # self.save_samples(name='test')

        else:
            assert self.mode is 'train', 'no test cfg'
            self.samples = self.load_samples(self.mode)
            self.sample_len = len(self.samples)


    def load_samples(self, name):
        with open('../cfg/Obman/{}.pkl'.format(name), 'rb') as f:
            samples = pickle.load(f)
            return samples

    def save_samples(self, name):
        with open('../cfg/Obman/{}.pkl'.format(name), 'wb') as f:
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
        img = self.transform(img)
        img = np.asarray(img.resize((416, 416), Image.ANTIALIAS), dtype=np.float32)
        if img.shape[-1] != 3:
            img = img[:, :, :-1]

        img = img / 255.
        # cv2.imshow("img in dataset", img)
        # cv2.waitKey(1)

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

    def fetch_image(self, sample):
        img_idx = sample['img_idx']

        hand_type = random.choices(population=['h+o', 'h'], weights=[self.ratio, 1 - self.ratio], k=1)[0]

        if hand_type is 'h+o':
            img = self.obman_dataset_all.get_image_BGR(img_idx)
        else:
            img = self.obman_dataset_hand.get_image_BGR(img_idx)
        return img


    def preprocess(self, idx):

        sample = self.samples[idx]

        handJoints3D = np.copy(sample['xyz'])
        handKps = handJoints3D[:, :-1]

        ### rescale keypoints first for extrapolation & augmentation ###
        handKps[:, 0] = handKps[:, 0] / self.downsample_ratio_x
        handKps[:, 1] = handKps[:, 1] / self.downsample_ratio_y

        ### Augmentation with GTs ###
        image = self.get_image(sample)

        # already processed color jitter, normalize to image
        if self.loadit and self.augment:
            Kps = handKps
            transformed = self.albumtransform(image=image, keypoints=Kps)

            image = transformed['image']
            handKps = np.array(transformed['keypoints'], dtype='float')

            ### random crop out image region ###
            transformed = self.albumtransform_img(image=image)
            image = transformed['image']

        image = np.squeeze(np.transpose(image, (2, 0, 1)))
        image = torch.from_numpy(image)
        if int(image.shape[0]) != 3:
            print("image shape wrong")
            image = image[:-1, :, :]
        image = self.normalize(image)

        ### hand pose ###
        del_u, del_v, del_z, cell = self.control_to_target(handKps, handJoints3D, self.loadit)  # handKps : [215.8, 182.1] , ...   /  handJoints3D : [~, ~, 462.2] , ...
        # hand pose tensor
        true_hand_pose = torch.zeros(21, 3, 5, 13, 13, dtype=torch.float32)
        u, v, z = cell
        pose = np.vstack((del_u, del_v, del_z)).T

        # try:
        true_hand_pose[:, :, z, v, u] = torch.from_numpy(pose)
        # except Exception as e:
        #     print(e)

        true_hand_pose = true_hand_pose.view(-1, 5, 13, 13)
        # hand mask
        hand_mask = torch.zeros(5, 13, 13, dtype=torch.float32)
        hand_mask[z, v, u] = 1

        return image, true_hand_pose, hand_mask  #, true_object_pose, object_mask, param_vis, vis, extra_hand_pose


    def downsample_points(self, points, depth):

        x = points[0]   # / self.downsample_ratio_x    ## already downsampled
        y = points[1]   # / self.downsample_ratio_y
        z = depth / 10.  # converting to centimeters

        downsampled_x = x / 32
        downsampled_y = y / 32
        downsampled_z = z / 25

        return downsampled_x, downsampled_y, downsampled_z

    def upsample_points(self, points, depth):
        u = points[0] * self.downsample_ratio_x
        v = points[1] * self.downsample_ratio_y
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

    def control_to_target(self, projected_points, points, loadit):
        # 2D location : projected_points[0, :]
        # depth : points[0, 2]
        root = projected_points[0, :]

        # get location of cell in (13, 13, 5)
        cell = self.get_cell(root, points[0, 2])
        if loadit:
            u, v, z = cell
            u = np.clip(u, 0, 12)
            v = np.clip(v, 0, 12)
            z = np.clip(z, 0, 4)
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
        y_hat = w_z * 25 * np.linalg.inv(self.camera_intrinsics).dot(points)

        return y_hat.T, points

    def target_to_control_wK(self, del_u, del_v, del_z, cell, K):

        u, v, z = cell

        w_u = del_u + u
        w_v = del_v + v
        w_z = del_z + z

        w_u, w_v, w_z = self.upsample_points((w_u, w_v), w_z)
        points = np.vstack((w_u * 32, w_v * 32, np.ones_like(w_u)))
        y_hat = w_z * 25 * np.linalg.inv(K).dot(points)

        return y_hat.T, points


if __name__ == '__main__':
    train = FHAD_Dataset(mode='test', loadit=False, name='test')
