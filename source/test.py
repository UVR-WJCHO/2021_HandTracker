import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append("./IKNet/")

import argparse
import time
import torch
import cv2
from tqdm import tqdm
import numpy as np
from cfg import parameters

from net import *
from dataset import HO3D_v2_Dataset, FreiHAND_Dataset
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

from scipy.spatial import procrustes


crop_threshold = 80


def _log_loss(args, init=False):
    if init:
        log_name = '../models/TEST_' + args['load_model'] + '.txt'
        with open(log_name, mode='wt', encoding='utf-8') as f:
            f.write('[Loss log]\n')
        return log_name


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-load_model", required=False, type=str, default='FCN_1111_res34_wfrei')
    ap.add_argument("-iknet", required=False, type=bool, default=False, help="activate iknet")
    ap.add_argument("-load_GT", required=False, type=bool, default=True, help="load GT extrapolation")
    ap.add_argument("-extra", required=False, type=bool, default=True,
                    help="activate extrapolation")  # action='store_true'

    ap.add_argument("-num_worker", required=False, type=int, default=1)
    ap.add_argument("-res34", required=False, type=bool, default=True, help="use res34 backbone")
    ap.add_argument("-lowerdim", required=False, type=bool, default=True, help="concatenate extra feature on lower part of network")
    ap.add_argument("-dataset", required=False, choices=['ho3d', 'frei'], default='frei', help="choose dataset option to train")

    args = vars(ap.parse_args())

    log_output_name = _log_loss(args, init=True)

    ############## Load dataset and set dataloader ##############
    if args['dataset'] is 'ho3d':
        testing_dataset = HO3D_v2_Dataset(mode='test', cfg='test', loadit=True, extra=args['extra'])
        # training_dataset = UnifiedPoseDataset(mode='train', loadit=True, name='train')
        testing_dataloader = torch.utils.data.DataLoader(testing_dataset, batch_size=1, shuffle=False, num_workers=args['num_worker'])
    if args['dataset'] is 'frei':
        testing_dataset = FreiHAND_Dataset(mode='test', loadit=True)
        # training_dataset = UnifiedPoseDataset(mode='train', loadit=True, name='train')
        testing_dataloader = torch.utils.data.DataLoader(testing_dataset, batch_size=1, shuffle=False,
                                                         num_workers=args['num_worker'])

    ############## Set model and training parameters ##############
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

    assert args['load_model'] is not None, 'need model name to load'

    model_load_name = '../models/' + args['load_model'] + '.pth'
    model.load_state_dict(torch.load(model_load_name), strict=False)
    model.eval()
    model.cuda()

    p_enc_3d = PositionalEncoding3D(21)
    z = torch.zeros((1, 13, 13, 5, 21))
    pos_encoder = p_enc_3d(z)

    downsample_ratio_x = 640./ 416.
    downsample_ratio_y = 480./ 416.

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

    if args['dataset'] is 'ho3d':
        print("...testing HO3D")
        model.update_parameter(img_width=640., img_height=480.)
        model.check_status()

    if args['dataset'] is 'frei':
        print("...testing FreiHand")
        model.update_parameter(img_width=224., img_height=224.)
        model.check_status()

    with torch.no_grad():
        total_loss = 0.
        err_3D_total = 0.
        err_2D_total = 0.
        err_pix_total = 0.
        err_z_total = 0.

        prev_handJoints3D_2 = None
        prev_handJoints3D_1 = None
        prev_handKps_2 = None
        prev_handKps_1 = None

        outlier_count = 0
        counting_idx = 0
        if args['dataset'] is 'ho3d':
            for batch, data in enumerate(tqdm(testing_dataloader)):
                image = data[0]
                if torch.isnan(image).any():
                    raise ValueError('Image error')
                true = [x.cuda() for x in data[1:-3]]
                crop_param = data[-1]
                [x_min, y_min, cropped_size] = np.array(crop_param)

                if args['load_GT']:
                    if args['extra']:
                        extra = data[-2]
                        pred = model(image.cuda(), extra.cuda())
                    else:
                        pred = model(image.cuda())

                else:
                    if args['extra']:
                        if data[-1]:
                            counting_idx = 0
                        if counting_idx < 2:
                            counting_idx += 1

                            extra_handKps = np.zeros((21, 2), dtype=np.float32)
                            extra_handJoints3D = np.zeros((21, 3), dtype=np.float32)
                        else:
                            root = prev_handKps_1[0, :] - prev_handKps_2[0, :]
                            dist = np.sqrt(root[0] * root[0] + root[1] * root[1])

                            if dist < 10.:
                                extra_handKps = np.copy(2 * prev_handKps_1 - prev_handKps_2)
                                extra_handJoints3D =  np.copy(2 * prev_handJoints3D_1 - prev_handJoints3D_2)
                            else:
                                extra_handKps = np.copy(prev_handKps_1)
                                extra_handJoints3D = np.copy(prev_handJoints3D_1)

                        extra_handKps[:, 0] = extra_handKps[:, 0] / downsample_ratio_x
                        extra_handKps[:, 1] = extra_handKps[:, 1] / downsample_ratio_y

                        del_u, del_v, del_z, cell = testing_dataset.control_to_target(extra_handKps, extra_handJoints3D, True)
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
                        pred = model(image.cuda())

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

                z, v, u = pred_hand_cell[1:]

                ### highest confidence value ###
                # highest_conf = pred_hand_conf[0, :, z, v, u]. reshape(21, 3)
                # print("highest conf value : ", 0)

                dels = pred_hand_pose[0, :, z, v, u].reshape(21, 3)
                del_u, del_v, del_z = dels[:, 0], dels[:, 1], dels[:, 2]
                hand_points, xy_points = testing_dataset.target_to_control(del_u, del_v, del_z, (u, v, z))
                ### currently xy_points on cropped-416 sized image

                # prev_handJoints3D_2 = np.copy(prev_handJoints3D_1)
                # prev_handKps_2 = np.copy(prev_handKps_1)
                #
                # prev_handJoints3D_1 = np.copy(hand_points)
                # prev_handKps_1 = np.transpose(np.copy(xy_points))[:, :-1]    # (21, 2)

                z, v, u = true_hand_cell[1:]
                dels_true = true_hand_pose[0, :, z, v, u].reshape(21, 3)
                del_u, del_v, del_z = dels_true[:, 0], dels_true[:, 1], dels_true[:, 2]
                true_hand_points, true_xy_points = testing_dataset.target_to_control(del_u, del_v, del_z, (u, v, z))

                xy_points = np.transpose(xy_points)[:, :-1]
                true_xy_points = np.transpose(true_xy_points)[:, :-1]

                ### recrop with crop_param
                # need to change crop_param torch to numpy
                # xy_points[:, 0] = (xy_points[:, 0] * (np.array(cropped_size)[0] / 416.)) + max(0, x_min - crop_threshold)
                # xy_points[:, 1] = (xy_points[:, 1] * (np.array(cropped_size)[1] / 416.)) + max(0, y_min - crop_threshold)
                #
                # true_xy_points[:, 0] = (true_xy_points[:, 0] * (cropped_size[0] / 416.)) + max(0, x_min - crop_threshold)
                # true_xy_points[:, 1] = (true_xy_points[:, 1] * (cropped_size[1] / 416.)) + max(0, y_min - crop_threshold)


                ### procrustes alignment ###
                #
                # mtx1, mtx2, disparity = procrustes(true_xy_points, xy_points)
                #
                # true_xy_points -= np.mean(true_xy_points, 0)
                # xy_points -= np.mean(xy_points, 0)
                #
                # norm1 = np.linalg.norm(true_xy_points)
                # norm2 = np.linalg.norm(xy_points)
                #
                # mtx1 *= norm1
                # mtx2 *= norm1

                ############################

                err_z = np.mean(((hand_points[:, -1] - true_hand_points[:, -1])) ** 2)
                err_pixel = np.mean((xy_points - true_xy_points) ** 2)  # (3, 21) - (3, 21) ... last raw is list of 1
                err_z = np.sqrt(err_z)
                err_pixel = np.sqrt(err_pixel)

                if err_pixel > 100:
                    outlier_count += 1
                    continue
                print("err : ", err_pixel, err_z)
                # pose_loss += mse_err
                err_pix_total += err_pixel
                err_z_total += err_z

                # print("err 2D : ", err_2D)
                #
                # vis_pred = pred_hand_vis[0, :, z, v, u].reshape(21, 1)
                # visible = []
                # for i in range(21):
                #     if vis_pred[i] > 0.5:
                #         visible.append(i)
                #
                img = testing_dataset.fetch_image(testing_dataset.samples[batch])
                # xy_points = true_xy_points

                # xy_points[:, 0] *= 640 / 416.
                # xy_points[:, 1] *= 480 / 416.
                # imgAnno = showHandJoints(img, xy_points)  # showHandJoints_vis(img, xy_points, visible)
                # imgAnno_rgb = imgAnno[:, :, [2, 1, 0]]
                # # imgAnno_rgb = cv2.flip(imgAnno_rgb, 1)
                # cv2.imshow("rgb pred", imgAnno_rgb)
                #
                # imgAnno_true = showHandJoints(img, true_xy_points)  # showHandJoints_vis(img, xy_points, visible)
                # imgAnno_rgb_true = imgAnno_true[:, :, [2, 1, 0]]
                # # imgAnno_rgb = cv2.flip(imgAnno_rgb, 1)
                # cv2.imshow("rgb true", imgAnno_rgb_true)
                # cv2.waitKey(0)

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
                    cv2.waitKey(0)

        if args['dataset'] is 'frei':
            for batch, data in enumerate(tqdm(testing_dataloader)):
                image = data[0]
                if torch.isnan(image).any():
                    raise ValueError('Image error')
                true = [x.cuda() for x in data[1:]]

                pred = model(image.cuda())

                loss = model.total_loss_FreiHAND(pred, true)
                total_loss += loss.data.detach().cpu().numpy()

                pred_hand_pose, pred_hand_conf, _, _, _ = [
                    p.data.cpu().numpy() for p in pred]
                true_hand_pose, hand_mask, K = [
                    t.data.cpu().numpy() for t in true]

                true_hand_cell = np.unravel_index(hand_mask.argmax(), hand_mask.shape)
                pred_hand_cell = np.unravel_index(pred_hand_conf.argmax(), pred_hand_conf.shape)

                z, v, u = pred_hand_cell[1:]
                dels = pred_hand_pose[0, :, z, v, u].reshape(21, 3)
                del_u, del_v, del_z = dels[:, 0], dels[:, 1], dels[:, 2]
                hand_points, xy_points = testing_dataset.target_to_control_wK(del_u, del_v, del_z, (u, v, z), K)

                z, v, u = true_hand_cell[1:]
                dels_true = true_hand_pose[0, :, z, v, u].reshape(21, 3)
                del_u, del_v, del_z = dels_true[:, 0], dels_true[:, 1], dels_true[:, 2]
                true_hand_points, true_xy_points = testing_dataset.target_to_control_wK(del_u, del_v, del_z,
                                                                                        (u, v, z), K)

                # crop_param
                # xy_points_debug = np.transpose(xy_points)
                # xy_points_debug[:, 2] = hand_points[:, 2]
                #
                # xy_points_debug[:, 0] = (xy_points_debug[:, 0] * (cropped_size[0] / 416.)) + max(0,
                #                                                                                  x_min - crop_threshold)
                # xy_points_debug[:, 1] = (xy_points_debug[:, 1] * (cropped_size[1] / 416.)) + max(0,
                #                                                                                  y_min - crop_threshold)

                xy_points = np.copy(np.transpose(xy_points))
                true_xy_points = np.copy(np.transpose(true_xy_points))

                ### procrustes alignment ###

                err_z = np.mean(((hand_points[:, -1] - true_hand_points[:, -1])) ** 2)
                err_pixel = np.mean((xy_points - true_xy_points) ** 2)  # (3, 21) - (3, 21) ... last raw is list of 1
                err_z = np.sqrt(err_z)
                err_pixel = np.sqrt(err_pixel)

                if err_pixel > 100:
                    outlier_count += 1
                    continue

                # pose_loss += mse_err
                err_pix_total += err_pixel
                err_z_total += err_z

                # print("err 2D : ", err_2D)
                #
                # vis_pred = pred_hand_vis[0, :, z, v, u].reshape(21, 1)
                # visible = []
                # for i in range(21):
                #     if vis_pred[i] > 0.5:
                #         visible.append(i)
                #
                # img = testing_dataset.fetch_image(testing_dataset.samples[batch])
                # # xy_points = true_xy_points
                #
                # # xy_points[:, 0] *= 640 / 416.
                # # xy_points[:, 1] *= 480 / 416.
                # imgAnno = showHandJoints(img, xy_points)  # showHandJoints_vis(img, xy_points, visible)
                # imgAnno_rgb = imgAnno[:, :, [2, 1, 0]]
                # # imgAnno_rgb = cv2.flip(imgAnno_rgb, 1)
                # cv2.imshow("rgb pred", imgAnno_rgb)
                # cv2.waitKey(0)

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
                    cv2.waitKey(0)

        with open(log_output_name, mode='at', encoding='utf-8') as f:
            log = total_loss * 1. / batch
            f.writelines(['Test Loss_FCN: ' + str(log) + '\n'])
            log = err_3D_total * 1. / (batch - outlier_count)
            f.writelines(['3D pose err : ' + str(log) + '\n'])
            log = err_2D_total * 1. / (batch - outlier_count)
            f.writelines(['2D pose err : ' + str(log) + '\n'])
            log = err_pix_total * 1. / (batch - outlier_count)
            f.writelines(['2D pixel err : ' + str(log) + '\n'])
            log = err_z_total * 1. / (batch - outlier_count)
            f.writelines(['z err : ' + str(log) + '\n'])
            log = outlier_count
            f.writelines(['outlier : ' + str(log)])