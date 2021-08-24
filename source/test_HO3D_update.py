import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append("./IKNet/")

import time
import torch
import cv2
from tqdm import tqdm
import numpy as np
from cfg import parameters

from net import UnifiedNetwork_update, UnifiedNetwork_v2
from dataset import UnifiedPoseDataset, HO3D_v2_Dataset, HO3D_v2_Dataset_update
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


if __name__ == '__main__':
    activate_extra = True
    flag_IKNet = False

    load_FCN_name = '../models/FCN_HO3D_0813_extra.pth'
    #load_IKNet_name = '../models/backup_iknet.pth'

    # dataset pkl are aligned
    # To shuffle the dataset w.r.t subject : set shuffle_seq=True
    # To shuffle the dataset totally : set shuffle=True in DataLoader
    testing_dataset_HO3D = HO3D_v2_Dataset_update(mode='test', cfg='test_align', loadit=True)
    # training_dataset = UnifiedPoseDataset(mode='train', loadit=True, name='train')
    testing_dataloader = torch.utils.data.DataLoader(testing_dataset_HO3D, batch_size=1,
                                                      shuffle=False,
                                                      num_workers=1)

    model = UnifiedNetwork_v2()
    model.load_state_dict(torch.load(load_FCN_name), strict=False)
    model.eval()
    model.cuda()

    p_enc_3d = PositionalEncoding3D(21)
    z = torch.zeros((1, 13, 13, 5, 21))
    pos_encoder = p_enc_3d(z)

    if flag_IKNet:
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

    with torch.no_grad():
        total_loss = 0.
        err_3D_total = 0.
        err_2D_total = 0.
        err_z_total = 0.

        prev_handJoints3D_2 = None
        prev_handJoints3D_1 = None
        prev_handKps_2 = None
        prev_handKps_1 = None

        outlier_count = 0
        counting_idx = 0
        for batch, data in enumerate(tqdm(testing_dataloader)):
            image = data[0]
            if torch.isnan(image).any():
                raise ValueError('Image error')
            true = [x.cuda() for x in data[1:-2]]
            flag_seq = data[-1]

            if flag_seq:
                counting_idx = 0

            if activate_extra:
                if counting_idx < 2:
                    counting_idx += 1

                    extra_handKps = np.zeros((21, 2), dtype=np.float32)
                    extra_handJoints3D = np.zeros((21, 3), dtype=np.float32)

                else:
                    extra_handKps = 2 * prev_handKps_1 - prev_handKps_2
                    extra_handJoints3D = 2 * prev_handJoints3D_1 - prev_handJoints3D_2

                del_u, del_v, del_z, cell = testing_dataset_HO3D.control_to_target(extra_handKps, extra_handJoints3D)
                # hand pose tensor
                # index + del, with positional encoding
                del_u = torch.unsqueeze(torch.from_numpy(del_u), 0).type(torch.float32)
                del_v = torch.unsqueeze(torch.from_numpy(del_v), 0).type(torch.float32)
                del_z = torch.unsqueeze(torch.from_numpy(del_z), 0).type(torch.float32)

                enc_cell = pos_encoder[:, cell[0], cell[1], cell[2], :].type(torch.float32)
                extra_hand_pose = torch.cat((enc_cell, del_u, del_v, del_z), 0)
                # extra_hand_pose = data[-1]

                pred = model(image.cuda(), extra_hand_pose.cuda())

            else:
                #t1 = time.time()
                pred = model(image.cuda())
                #print("time : ", time.time() - t1)

            loss = model.total_loss(pred, true)
            total_loss += loss.data.cpu().numpy()

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

            prev_handJoints3D_2 = prev_handJoints3D_1
            prev_handKps_2 = prev_handKps_1

            prev_handJoints3D_1 = hand_points
            prev_handKps_1 = np.transpose(xy_points)[:, :-1]    # (21, 2)

            dels_true = true_hand_pose[0, :, z, v, u].reshape(21, 3)
            del_u, del_v, del_z = dels_true[:, 0], dels_true[:, 1], dels_true[:, 2]
            true_hand_points, true_xy_points = testing_dataset_HO3D.target_to_control(del_u, del_v, del_z, (u, v, z))

            err_z = np.mean(((hand_points[:, -1] - true_hand_points[:, -1])) ** 2)
            err_2D = np.mean((xy_points - true_xy_points) ** 2) # (3, 21) - (3, 21) ... last raw is list of 1

            err_3D = np.sqrt(err_z + err_2D)
            err_z = np.sqrt(err_z)
            err_2D = np.sqrt(err_2D)

            if err_2D > 100:
                outlier_count += 1
                continue

            #pose_loss += mse_err
            err_3D_total += err_3D
            err_2D_total += err_2D
            err_z_total += err_z

            # print("err 2D : ", err_2D)
            #
            # vis_pred = pred_hand_vis[0, :, z, v, u].reshape(21, 1)
            # visible = []
            # for i in range(21):
            #     if vis_pred[i] > 0.5:
            #         visible.append(i)
            #
            # img = testing_dataset_HO3D.fetch_image(testing_dataset_HO3D.samples[batch])
            #
            # xy_points = np.transpose(xy_points)
            #
            # imgAnno = showHandJoints_vis(img, xy_points, visible)   # imgAnno = showHandJoints(img, xy_points)
            #
            # imgAnno_rgb = imgAnno[:, :, [2, 1, 0]]
            # imgAnno_rgb = cv2.flip(imgAnno_rgb, 1)
            # cv2.imshow("rgb pred", imgAnno_rgb)
            # cv2.waitKey(1)

            ############################ IKNet ############################
            if flag_IKNet:
                root = hand_points[9, :]
                hand_points_rel = (root - hand_points) / 100.

                # check 'hand_points' device
                xyz = torch.from_numpy(hand_points_rel).cuda()
                _, theta_mpii = IKNet(xyz.float())

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
                cv2.waitKey(0)

        print(total_loss * 1. / batch)
        #print("mse pose err(missing scaling) : ", pose_loss * 1. / (batch - outlier_count))
        print("3D pose err : ", err_3D_total * 1. / (batch - outlier_count))
        print("2D pose err : ", err_2D_total * 1. / (batch - outlier_count))
        print("outlier : ", outlier_count)