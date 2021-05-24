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
from net import UnifiedNetwork
from dataset import UnifiedPoseDataset
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


class extrapolation():
    def __init__(self):
        self.prev_idx = '-1'
        self.prev_batch_idx = '-1'

        self.prev_gt_3d = torch.zeros([2, 22, 3], dtype=torch.float32)

    def grid_to_3d(self, curr_gt, hand_mask_list, batch_len):
        for i in range(batch_len):
            hand_mask = hand_mask_list[i].unsqueeze(0)
            true_hand_cell = np.unravel_index(hand_mask.argmax(), hand_mask.shape)
            z, v, u = true_hand_cell[1:]
            dels = curr_gt[0, :, z, v, u].reshape(21, 3)
            cell = torch.FloatTensor([z, v, u])  # (3)
            dels = torch.cat([cell.unsqueeze(0), dels], dim=0)
            if i == 0:
                self.curr_gt_3d = dels.unsqueeze(0)
            else:
                self.curr_gt_3d = torch.cat([self.curr_gt_3d, dels.unsqueeze(0)], dim=0)

        self.stacked_gt = torch.cat([self.prev_gt_3d, self.curr_gt_3d], dim=0).cuda()
        self.prev_gt_3d = self.curr_gt_3d[-2:, ]

        return self.stacked_gt

    def extrapolate(self, batch_len, seq_idx):
        flag_pass = False
        for i in range(batch_len):
            curr_idx = seq_idx[i]
            if flag_pass:
                flag_pass = False
                self.prev_idx = curr_idx
                continue

            if i == 0:
                if curr_idx != self.prev_batch_idx:
                    extra = torch.zeros([2, 22, 3], dtype=torch.float32).cuda()
                    flag_pass = True
                else:
                    extra = (2 * self.stacked_gt[1] - self.stacked_gt[0]).unsqueeze(0)
            else:
                if curr_idx != self.prev_idx:
                    if i != (batch_len - 1):
                        ex = torch.zeros([2, 22, 3], dtype=torch.float32).cuda()
                    else:
                        ex = torch.zeros([1, 22, 3], dtype=torch.float32).cuda()
                    extra = torch.cat([extra, ex], dim=0)
                    flag_pass = True

                else:
                    ex = 2 * self.stacked_gt[i + 1] - self.stacked_gt[i]
                    extra = torch.cat([extra, ex.unsqueeze(0)], dim=0)

            self.prev_idx = curr_idx

            self.prev_batch_idx = self.prev_idx

        return extra


if __name__ == '__main__':
    flag_extra = False

    load_FCN_name = '../models/FCN_base.pth'
    load_IKNet_name = '../models/backup_iknet.pth'

    training_dataset = UnifiedPoseDataset(mode='test', loadit=True, name='test')
    training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size = 1, shuffle=False, num_workers=1)

    extp_module = extrapolation()

    model = UnifiedNetwork()
    model.load_state_dict(torch.load(load_FCN_name))
    model.eval()
    model.cuda()

    IKNet = minimal_hand(config.HAND_MESH_MODEL_PATH, load_IKNet_name)
    IKNet.cuda()
    IKNet.eval()

    ### Minimal hand ###
    render = o3d_render(config.HAND_MESH_MODEL_PATH)
    extrinsic = render.extrinsic
    extrinsic[0:3, 3] = 0
    render.extrinsic = extrinsic
    render.intrinsic = [config.CAM_FX, config.CAM_FY]
    render.updata_params()
    render.environments('./IKNet/render_option.json', 1000)

    mesh_smoother = OneEuroFilter(4.0, 0.0)
    clock = pygame.time.Clock()

    #cap = cv2.VideoCapture(0)

    # validation
    with torch.no_grad():
        hand_cell_counter = 0.
        object_cell_counter = 0.
        object_counter = 0.
        action_counter = 0.

        hand_detected = False
        object_detected = False

        loss_bfIK = 0.
        loss_afIK = 0.
        outlier_count = 0
        for batch, data in enumerate(tqdm(training_dataloader)):
            #print(training_dataset.samples[batch])
            image = data[0]
            true = [x.cuda() for x in data[1:-1]]

            # original_img = np.squeeze(image.cpu().numpy())
            # original_img = np.transpose(original_img, (1, 2, 0))
            # original_img = original_img.copy()[..., ::-1]
            # original_img = cv2.resize(original_img, (256, 256))
            ### From camera input ###

            # _, frame_large = cap.read()
            # frame_large = np.flip(frame_large, -1).copy()
            # frame_large = np.flip(frame_large, axis=1).copy()
            # frame_large = frame_large[:, 80:-80, :]
            # frame = imresize(frame_large, (128, 128))
            #
            # original_img = frame.copy()[..., ::-1]
            # original_img = cv2.resize(original_img, (256, 256))
            #
            # frame = torch.from_numpy(frame)
            # frame = rearrange(frame, 'h w c -> 1 c h w')
            # frame = frame.float() / 255
            # image = frame

            ############################ FCN ############################
            if flag_extra:
                print("...")
            else:
                pred = model(image.cuda())
                loss = model.total_loss(pred, true)

            ############################ IKNet ############################
            if flag_extra:
                print("...")
            else:
                hand_points_rel, true_hand_points_rel, root, hp, thp = IKNet.extract_handkeypoint(pred, true)
                xyz = torch.tensor(hand_points_rel, requires_grad=True).cuda()
                xy_points, theta_mpii = IKNet(xyz.float())

            xy_points_world = root - xy_points.cpu().numpy() * 100.

            bfIK = np.sqrt(np.mean((hp - thp) ** 2))
            afIK = np.sqrt(np.mean((xy_points_world - thp) ** 2))

            if bfIK > 100 or afIK > 100:
                outlier_count += 1
                continue

            loss_bfIK += bfIK
            loss_afIK += afIK
            # rendering
            theta_mpii = theta_mpii.detach().cpu().numpy()
            theta_mano = mpii_to_mano(np.squeeze(theta_mpii))

            v = render.hand_mesh.set_abs_quat(theta_mano)
            v *= 2  # for better visualization
            v = v * 1000 + np.array([0, 0, 400])
            v = mesh_smoother.process(v)

            render.rendering(v, config.HAND_COLOR)
            render_img = render.capture_img()
            render_img = cv2.resize(render_img, (256, 256))
            #save_img = np.concatenate([original_img, render_img], axis=1)

            xyz_FK = render.hand_mesh.set_abs_xyz(theta_mano)
            xyz_ori = np.squeeze(xyz.cpu().numpy())
            xyz_ori = mpii_to_mano(xyz_ori)# * 10.0

            cv2.imshow("result", render_img)
            cv2.waitKey(0)
            


            ################# original image visualization ###################
            # xy_points = np.squeeze(xy_points.cpu().numpy())
            # handKps = xy_points[:, :-1]
            #
            # img = training_dataset.fetch_image(training_dataset.samples[batch])
            # imgAnno = showHandJoints(img, handKps)
            # imgAnno_rgb = imgAnno[:, :, [2, 1, 0]]
            # imgAnno_rgb = cv2.flip(imgAnno_rgb, 1)
            # cv2.imshow("rgb pred", imgAnno_rgb)
            # cv2.waitKey(0)

        print(loss_bfIK * 1. / (batch - outlier_count))
        print(loss_afIK * 1. / (batch - outlier_count))
        print("outlier : ", outlier_count)
        # print(hand_cell_counter * 1. / batch)
        # print(object_cell_counter * 1. / batch)
        # print(action_counter * 1 / hand_cell_counter)
        # print(object_counter * 1 / object_cell_counter)
