import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append("./IKNet/")

from tqdm import tqdm
import torch
import numpy as np
import time
from cfg import parameters
from net import UnifiedNetwork_update
from dataset import UnifiedPoseDataset, HO3D_v2_Dataset
from visualize import UnifiedVisualization
from vis_utils.vis_utils import *
import cv2
import pygame
import IKNet.config as config
from IKNet.hand_mesh import HandMesh
from IKNet.kinematics import *
from IKNet.utils import *
from IKNet.model.hand_mesh import minimal_hand
from einops import rearrange
from open3d import io as io
from IKNet.render import o3d_render
from IKNet.capture import OpenCVCapture
from PIL import Image


if __name__ == '__main__':
    flag_extra = False
    flag_suffle = False

    load_FCN_name = '../models/FCN_HO3D_trial_1.pth'
    #load_IKNet_name = '../models/backup_iknet.pth'

    testing_dataset_HO3D = HO3D_v2_Dataset(mode='test', loadit=True, shuffle=flag_suffle)
    # training_dataset = UnifiedPoseDataset(mode='train', loadit=True, name='train')
    testing_dataloader = torch.utils.data.DataLoader(testing_dataset_HO3D, batch_size=1,
                                                      shuffle=False,
                                                      num_workers=1)

    model = UnifiedNetwork_update()
    model.load_state_dict(torch.load(load_FCN_name), strict=False)
    model.eval()
    model.cuda()

    with torch.no_grad():
        total_loss = 0
        for batch, data in enumerate(tqdm(testing_dataloader)):
            image = data[0]
            if torch.isnan(image).any():
                raise ValueError('Image error')
            true = [x.cuda() for x in data[1:]]

            t1 = time.time()
            pred = model(image.cuda())
            print("time : ", time.time() - t1)
            loss = model.total_loss(pred, true)

            total_loss += loss

            pred_hand_pose, pred_hand_conf, pred_object_pose, pred_object_conf, pred_hand_vis = [
                p.data.cpu().numpy() for p in pred]
            true_hand_pose, hand_mask, true_object_pose, object_mask, true_hand_vis = [
                t.data.cpu().numpy() for t in true]

            true_hand_cell = np.unravel_index(hand_mask.argmax(), hand_mask.shape)
            true_object_cell = np.unravel_index(object_mask.argmax(), object_mask.shape)

            pred_hand_cell = np.unravel_index(pred_hand_conf.argmax(), pred_hand_conf.shape)
            pred_object_cell = np.unravel_index(pred_object_conf.argmax(), pred_object_conf.shape)

            z, v, u = true_hand_cell[1:]
            dels = pred_hand_pose[0, :, z, v, u].reshape(21, 3)
            del_u, del_v, del_z = dels[:, 0], dels[:, 1], dels[:, 2]
            hand_points, xy_points = testing_dataset_HO3D.target_to_control(del_u, del_v, del_z, (u, v, z))

            vis_pred = pred_hand_vis[0, :, z, v, u].reshape(21, 1)

            img = testing_dataset_HO3D.fetch_image(testing_dataset_HO3D.samples[batch])

            xy_points = np.transpose(xy_points)
            imgAnno = showHandJoints(img, xy_points)
            imgAnno_rgb = imgAnno[:, :, [2, 1, 0]]
            imgAnno_rgb = cv2.flip(imgAnno_rgb, 1)
            cv2.imshow("rgb pred", imgAnno_rgb)
            cv2.waitKey(0)

        print(total_loss * 1. / batch)