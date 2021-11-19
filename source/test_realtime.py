import os, inspect
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append("./IKNet/")
currentdir=os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
basedir=currentdir+'/../'
sys.path.append(basedir+'sensor/')

import argparse
import time
import torch
import cv2
from tqdm import tqdm
import numpy as np
from cfg import parameters

from net import *
from dataset import HO3D_v2_Dataset
from vis_utils.vis_utils import *

import IKNet.config as config
from IKNet.hand_mesh import HandMesh
from IKNet.kinematics import *
from IKNet.utils import *
from IKNet.model.hand_mesh import minimal_hand
from IKNet.render import o3d_render
from IKNet.capture import OpenCVCapture

# from einops import rearrange
# from open3d import io as io
# from PIL import Image
from positional_encodings import PositionalEncoding1D, PositionalEncoding2D, PositionalEncoding3D
from torchvision import transforms
import pyrealsense2 as rs
from Realsense import Realsense
import matplotlib.pyplot as plt


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-load_model", required=False, type=str, default='FCN_1104_Frei_init_150epoch+80')
    ap.add_argument("-iknet", required=False, type=bool, default=True, help="activate iknet")
    ap.add_argument("-extra", required=False, type=bool, default=False, help="activate extrapolation")  # action='store_true'

    ap.add_argument("-res34", required=False, type=bool, default=False, help="use res34 backbone")
    ap.add_argument("-lowerdim", required=False, type=bool, default=True, help="concatenate extra feature on lower part of network")

    args = vars(ap.parse_args())

    testing_dataset_HO3D = HO3D_v2_Dataset(mode='test', cfg='test', loadit=True, extra=args['extra'])
    downsample_ratio_x = 640. / 416.
    downsample_ratio_y = 480. / 416.
    ############## Set model and training parameters ##############
    assert (torch.cuda.is_available())
    device = torch.device('cuda:0')

    if args['res34']:
        model = UnifiedNet_res34()
        if args['lowerdim']:
            model = UnifiedNet_res34_lowconcat()
    else:
        if not args['extra']:
            model = UnifiedNet_res18_noextra()
            print("...testing without extra")
        else:
            model = UnifiedNet_res18()
            if args['lowerdim']:
                model = UnifiedNet_res18_lowconcat()

    model_load_name = '../models/' + args['load_model'] + '.pth'
    model.load_state_dict(torch.load(model_load_name), strict=False)
    model.eval()
    model.cuda()

    p_enc_3d = PositionalEncoding3D(21)
    z = torch.zeros((1, 13, 13, 5, 21))
    pos_encoder = p_enc_3d(z)

    transform = transforms.ColorJitter(0.5, 0.5, 0.5)
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    if args['iknet']:
        render = o3d_render(config.HAND_MESH_MODEL_PATH)
        extrinsic = render.extrinsic
        extrinsic[0:3, 3] = 0
        render.extrinsic = extrinsic
        render.intrinsic = [config.CAM_FX, config.CAM_FY]
        render.updata_params()
        render.environments('./IKNet/render_option.json', 1000)
        mesh_smoother = OneEuroFilter(4.0, 0.0)

        IKNet = minimal_hand(config.HAND_MESH_MODEL_PATH, '../models/IKNet.pth')
        # hand_machine = minimal_hand(config.HAND_MESH_MODEL_PATH, './IKNet/weights/detnet.pth', './IKNet/weights/iknet.pth')
        IKNet.cuda()
        IKNet.eval()

    ############## Camera setting ##############
    save_enabled = False
    frame_saved = 0
    frame = 0
    save_filepath = './save_pth/'
    realsense = Realsense.Realsense()

    with torch.no_grad():
        prev_handJoints3D_2 = None
        prev_handJoints3D_1 = None
        prev_handKps_2 = None
        prev_handKps_1 = None

        outlier_count = 0
        counting_idx = 0

        try:
            while True:
                realsense.run()

                depth = realsense.getDepthImage()
                color = realsense.getColorImage()

                depth = depth.astype(np.uint16)

                depth_seg = depth.copy()
                depth_seg[depth_seg > 500] = 0
                depth_norm = depth_seg / np.max(depth_seg)

                # # cv2.imshow('depth',np.uint8(depth))
                # cv2.imshow('depth_seg', np.uint8(depth_seg))
                # cv2.imshow('depth_seg 2', depth_norm)

                color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
                image = cv2.resize(color, dsize=(416, 416), interpolation=cv2.INTER_CUBIC)
                image = image / 255.

                cv2.imshow('color', color)
                cv2.waitKey(1)

                image = np.squeeze(np.transpose(image, (2, 0, 1)))
                image = torch.from_numpy(image)
                if int(image.shape[0]) != 3:
                    print("image shape wrong")
                    image = image[:-1, :, :]
                image = normalize(image)
                image = torch.unsqueeze(image, 0).type(torch.float32)
                if not args['extra']:
                    pred = model(image.cuda())
                else:
                    if counting_idx < 2:
                        counting_idx += 1
                        extra_handKps = np.zeros((21, 2), dtype=np.float32)
                        extra_handJoints3D = np.zeros((21, 3), dtype=np.float32)
                    else:
                        root = prev_handKps_1[0, :] - prev_handKps_2[0, :]
                        dist = np.sqrt(root[0] * root[0] + root[1] * root[1])

                        if dist < 10.:
                            extra_handKps = np.copy(2 * prev_handKps_1 - prev_handKps_2)
                            extra_handJoints3D = np.copy(2 * prev_handJoints3D_1 - prev_handJoints3D_2)
                        else:
                            extra_handKps = np.copy(prev_handKps_1)
                            extra_handJoints3D = np.copy(prev_handJoints3D_1)

                    extra_handKps[:, 0] = extra_handKps[:, 0] / downsample_ratio_x
                    extra_handKps[:, 1] = extra_handKps[:, 1] / downsample_ratio_y

                    del_u, del_v, del_z, cell = testing_dataset_HO3D.control_to_target(extra_handKps,
                                                                                       extra_handJoints3D, True)
                    # hand pose tensor
                    # index + del, with positional encoding
                    del_u = torch.unsqueeze(torch.from_numpy(del_u), 0).type(torch.float32)
                    del_v = torch.unsqueeze(torch.from_numpy(del_v), 0).type(torch.float32)
                    del_z = torch.unsqueeze(torch.from_numpy(del_z), 0).type(torch.float32)

                    enc_cell = pos_encoder[:, cell[0], cell[1], cell[2], :].type(torch.float32)
                    extra_hand_pose = torch.cat((enc_cell, del_u, del_v, del_z), 0)
                    # extra_hand_pose = data[-1]

                    pred = model(image.cuda(), extra_hand_pose.cuda())

                pred_hand_pose, pred_hand_conf, pred_object_pose, pred_object_conf, pred_hand_vis = [
                    p.data.cpu().numpy() for p in pred]

                pred_hand_cell = np.unravel_index(pred_hand_conf.argmax(), pred_hand_conf.shape)
                pred_object_cell = np.unravel_index(pred_object_conf.argmax(), pred_object_conf.shape)

                z, v, u = pred_hand_cell[1:]

                dels = pred_hand_pose[0, :, z, v, u].reshape(21, 3)
                del_u, del_v, del_z = dels[:, 0], dels[:, 1], dels[:, 2]
                hand_points, xy_points = testing_dataset_HO3D.target_to_control(del_u, del_v, del_z, (u, v, z))

                prev_handJoints3D_2 = np.copy(prev_handJoints3D_1)
                prev_handKps_2 = np.copy(prev_handKps_1)

                prev_handJoints3D_1 = np.copy(hand_points)
                prev_handKps_1 = np.transpose(np.copy(xy_points))[:, :-1]  # (21, 2)

                xy_points = np.transpose(xy_points)
                # xy_points[:, 0] *= 640 / 416.
                # xy_points[:, 1] *= 480 / 416.
                imgAnno = showHandJoints(color, xy_points)  # showHandJoints_vis(img, xy_points, visible)
                imgAnno_rgb = imgAnno[:, :, [2, 1, 0]]
                # imgAnno_rgb = cv2.flip(imgAnno_rgb, 1)
                cv2.imshow("rgb pred", imgAnno_rgb)
                cv2.waitKey(1)

                # save original image
                if save_enabled is True and frame > -1:
                    cv2.imwrite(save_filepath + 'depth_%d.png' % frame_saved, depth)
                    cv2.imwrite(save_filepath + 'color_%d.png' % frame_saved, color)
                    frame_saved += 1


                frame += 1
                if frame % 100 is 0:
                    print('frame..', frame)

                ############################ IKNet ############################
                if args['iknet']:
                    root = hand_points[9, :]
                    hand_points_rel = (root - hand_points) / 100.
                    hand_points_rel[:, 0] *= -1
                    hand_points_rel[:, 2] *= -1

                    # check 'hand_points' device
                    xyz = torch.from_numpy(hand_points_rel).cuda()
                    t_a = time.time()
                    _, theta_mpii = IKNet(xyz.float())
                    print("time IK : ", time.time() - t_a)
                    theta_mpii = theta_mpii.detach().cpu().numpy()
                    theta_mano = mpii_to_mano(np.squeeze(theta_mpii))

                    v = render.hand_mesh.set_abs_quat(theta_mano)
                    v *= 2  # for better visualization
                    v = v * 1000 + np.array([0, 0, 400])
                    v = mesh_smoother.process(v)

                    render.rendering(v, config.HAND_COLOR)
                    render_img = render.capture_img()
                    render_img = cv2.resize(render_img, (256, 256))
                    # save_img = np.concatenate([original_img, render_img], axis=1)

                    xyz_FK = render.hand_mesh.set_abs_xyz(theta_mano)
                    xyz_ori = np.squeeze(xyz.cpu().numpy())
                    xyz_ori = mpii_to_mano(xyz_ori)  # * 10.0

                    cv2.imshow("result", render_img)
                    cv2.waitKey(1)


        finally:
            print('stop device')
            realsense.release()
            print(frame)


