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
from IKNet.kinematics import mpii_to_mano
from IKNet.utils import *
from IKNet.model.hand_mesh import minimal_hand
from einops import rearrange
from open3d import io as io
from IKNet.render import o3d_render
from IKNet.capture import OpenCVCapture
from PIL import Image


def extract_handkeypoint(pred, true):
    pred_hand_pose, pred_action_prob, pred_hand_conf, pred_object_pose, pred_object_prob, pred_object_conf = [
        p.data.cpu().numpy() for p in pred]
    true_hand_pose, true_action_prob, hand_mask, true_object_pose, true_object_prob, object_mask = [t.data.cpu().numpy()
                                                                                                    for t in true]

    true_hand_cell = np.unravel_index(hand_mask.argmax(), hand_mask.shape)

    z, v, u = true_hand_cell[1:]
    dels = pred_hand_pose[0, :, z, v, u].reshape(21, 3)
    del_u, del_v, del_z = dels[:, 0], dels[:, 1], dels[:, 2]
    hand_points, xy_points = training_dataset.target_to_control(del_u, del_v, del_z, (u, v, z))

    return hand_points, xy_points


if __name__ == '__main__':
    flag_detNet = False

    training_dataset = UnifiedPoseDataset(mode='test', loadit=True, name='test')
    training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size = 1, shuffle=False, num_workers=1)

    model = UnifiedNetwork()
    model.load_state_dict(torch.load('../models/FCN.pth'))
    model.eval()
    model.cuda()

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

    IKNet = minimal_hand(config.HAND_MESH_MODEL_PATH, '../models/IKNet.pth')
    # hand_machine = minimal_hand(config.HAND_MESH_MODEL_PATH, './IKNet/weights/detnet.pth', './IKNet/weights/iknet.pth')
    IKNet.cuda()
    IKNet.eval()

    cap = cv2.VideoCapture(0)



    # validation
    with torch.no_grad():

        hand_cell_counter = 0.
        object_cell_counter = 0.
        object_counter = 0.
        action_counter = 0.

        hand_detected = False
        object_detected = False

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
            if not flag_detNet:
                pred = model(image.cuda())
                loss = model.total_loss(pred, true)
            """
            pred_hand_pose, pred_action_prob, pred_hand_conf, pred_object_pose, pred_object_prob, pred_object_conf = [p.data.cpu().numpy() for p in pred]
            true_hand_pose, true_action_prob, hand_mask, true_object_pose, true_object_prob, object_mask = [t.data.cpu().numpy() for t in true]

            true_hand_cell = np.unravel_index(hand_mask.argmax(), hand_mask.shape)
            true_object_cell = np.unravel_index(object_mask.argmax(), object_mask.shape)

            pred_hand_cell = np.unravel_index(pred_hand_conf.argmax(), pred_hand_conf.shape)
            pred_object_cell = np.unravel_index(pred_object_conf.argmax(), pred_object_conf.shape)

            # hand cell correctly detected
            hand_cell_counter += int(true_hand_cell == pred_hand_cell)
            hand_detected = true_hand_cell == pred_hand_cell

            # object cell correctly detected
            object_cell_counter += int(true_object_cell == pred_object_cell)
            object_detected = true_object_cell == pred_object_cell

            z, v, u = true_hand_cell[1:]
            dels = pred_hand_pose[0,:,z, v, u].reshape(21, 3)
            del_u, del_v, del_z = dels[:,0], dels[:,1], dels[:,2]
            hand_points, xy_points = training_dataset.target_to_control(del_u, del_v, del_z, (u, v, z))

            if hand_detected:
                action_counter += int(np.argmax(pred_action_prob[0, :, z, v, u]) == true_action_prob[0, z, v, u])
        
            z, v, u = true_object_cell[1:]
            dels = pred_object_pose[0,:,z, v, u].reshape(21, 3)
            del_u, del_v, del_z = dels[:,0], dels[:,1], dels[:,2]
            object_points, _ = training_dataset.target_to_control(del_u, del_v, del_z, (u, v, z))

            if object_detected:
                object_counter += int(np.argmax(pred_object_prob[0, :, z, v, u]) == true_object_prob[0, z, v, u])
            """
            ############################ IKNet ############################
            if not flag_detNet:
                hand_points, xy_points = extract_handkeypoint(pred, true)

                # FCN for right hand, IKNet initial for Left hand
                #hand_points[:, 1] = 256 - hand_points[:,1]

                root = hand_points[9, :]
                hand_points_rel = (root - hand_points) / 100.

                # check 'hand_points' device
                xyz = torch.from_numpy(hand_points_rel).cuda()
                _, theta_mpii = IKNet(xyz.float())
            else:
                image = image.cuda()
                xyz, theta_mpii = IKNet(image)

            theta_mpii = theta_mpii.detach().cpu().numpy()
            theta_mano = mpii_to_mano(np.squeeze(theta_mpii))

            # rendering

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
            


            ####################################
            if not flag_detNet:
                xy_points = np.transpose(xy_points)
                handKps = xy_points[:, :-1]

                img = training_dataset.fetch_image(training_dataset.samples[batch])
                imgAnno = showHandJoints(img, handKps)
                imgAnno_rgb = imgAnno[:, :, [2, 1, 0]]
                imgAnno_rgb = cv2.flip(imgAnno_rgb, 1)
                cv2.imshow("rgb pred", imgAnno_rgb)
                cv2.waitKey(0)


        print(hand_cell_counter * 1. / batch)
        print(object_cell_counter * 1. / batch)
        print(action_counter * 1 / hand_cell_counter)
        print(object_counter * 1 / object_cell_counter)
