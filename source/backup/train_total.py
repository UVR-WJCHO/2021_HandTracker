import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append("./IKNet")

import random
from tqdm import tqdm
import torch
from cfg import parameters
from net import UnifiedNetwork
from dataset import UnifiedPoseDataset
import numpy as np
from tensorboardX import SummaryWriter
import time
from vis_utils.vis_utils import *

import IKNet.config as config
from IKNet.utils import *
from IKNet.model.hand_mesh import minimal_hand
from IKNet.render import o3d_render
from IKNet.kinematics import *

from IKNet.hand_mesh import HandMesh
from IKNet.kinematics import mpii_to_mano
from einops import rearrange
from open3d import io as io
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

    dataset_suffle = True

    flag_extra = True
    if flag_extra:
        dataset_suffle = False
        parameters.epochs -= 20

    continue_train = False
    load_epoch = 0

    model_FCN_name = '../models/FCN_M1.pth'
    model_IKNet_name = '../models/IKNet_M1.pth'
    HAND_MESH_MODEL_PATH = './IKNet/IKmodel/hand_mesh/hand_mesh_model.pkl'

    training_dataset = UnifiedPoseDataset(mode='train', loadit=True, name='train')
    testing_dataset = UnifiedPoseDataset(mode='test', loadit=True, name='test')

    training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=parameters.batch_size, shuffle=dataset_suffle, num_workers=4)
    testing_dataloader = torch.utils.data.DataLoader(testing_dataset, batch_size=parameters.batch_size, shuffle=False, num_workers=4)

    model = UnifiedNetwork()
    model.load_state_dict(torch.load('../models/FCN_init.pth'))
    IKNet = minimal_hand(config.HAND_MESH_MODEL_PATH, '../models/IKNet_init.pth')

    model.cuda()
    IKNet.cuda()

    extp = extrapolation()

    render = o3d_render(config.HAND_MESH_MODEL_PATH)
    extrinsic = render.extrinsic
    extrinsic[0:3, 3] = 0
    render.extrinsic = extrinsic
    render.intrinsic = [config.CAM_FX, config.CAM_FY]
    render.updata_params()
    render.environments('./IKNet/render_option.json', 1000)

    assert (torch.cuda.is_available())

    if continue_train:
        device = torch.device('cuda:0')
        state_dict = torch.load(model_FCN_name, map_location=str(device))
        model.load_state_dict(state_dict)

        state_dict = torch.load(model_IKNet_name, map_location=str(device))
        IKNet.load_state_dict(state_dict)
        print("load success")

    lr_FCN = 0.0001
    lr_IKN = 0.0001

    optimizer_FCN = torch.optim.Adam(model.parameters(), lr=lr_FCN)
    optimizer_IKN = torch.optim.Adam(IKNet.parameters(), lr=lr_IKN)

    best_loss = float('inf')
    #writer = SummaryWriter()

    epoch_range = parameters.epochs
    if continue_train:
        epoch_range -= load_epoch

    for epoch in range(epoch_range):
        # train
        model.train()
        training_loss_FCN = 0.
        training_loss_IK = 0.

        training_dataset.randomize_order()

        for batch, data in enumerate(tqdm(training_dataloader)):

            #t1 = time.time()
            optimizer_FCN.zero_grad()
            optimizer_IKN.zero_grad()
            image = data[0]
            if torch.isnan(image).any():
                raise ValueError('Image error')

            true = [x.cuda() for x in data[1:-1]]   # added sequence index in dataset

            ############################ FCN ############################
            if flag_extra:
                seq_idx = data[-1]  # tupple, string

                curr_gt = data[1]
                hand_mask_list = data[3]
                batch_len = hand_mask_list.shape[0]
                stacked_gt = extp.grid_to_3d(curr_gt, hand_mask_list, batch_len).cuda()
                # stacked_gt : torch, torch.Size([batchsize+2, 22, 3]), gpu

                # At initial or if training dataset's sequence # changed, apply zero extra_keypoint
                # if not, extrapolate keypoint from previous pred
                extra = extp.extrapolate(batch_len, seq_idx)
                # extra : (batch_size, 22, 3)

                pred = model(image.cuda(), extra)
                loss = model.total_loss(pred, true)
                training_loss_FCN += loss.data.cpu().numpy()

            else:
                pred = model(image.cuda())
                loss = model.total_loss(pred, true)

                training_loss_FCN += loss.data.cpu().numpy()
            #t2 = time.time()
            ############################ IKNet ############################
            if flag_extra:
                hand_points_rel, true_hand_points_rel = IKNet.extract_handkeypoint_batch(pred, true)
                xyz = torch.tensor(hand_points_rel, requires_grad=True).cuda()
                _, theta_mpii_batch = IKNet(xyz.float())
                loss_FK = 0
                # theta_mpii = theta_mpii.cpu().numpy()
                batch_len = theta_mpii_batch.shape[0]
                for i in range(batch_len):
                    theta_mpii = theta_mpii_batch[i]
                    theta_mano = mpii_to_mano_torch(theta_mpii)

                    xyz_FK = render.hand_mesh.set_abs_xyz_torch(theta_mano) * 10.0
                    # xyz_FK : mano order
                    xyz_ori = xyz[i].type(torch.FloatTensor).cuda()
                    xyz_ori = mpii_to_mano_torch(xyz_ori)

                    loss_FK += IKNet.ForwardKinematic_loss(xyz_FK, xyz_ori)
            else:
                hand_points_rel, true_hand_points_rel = IKNet.extract_handkeypoint_batch(pred, true)

                xyz = torch.tensor(hand_points_rel, requires_grad=True).cuda()

                _, theta_mpii_batch = IKNet(xyz.float())

                loss_FK = 0
                #theta_mpii = theta_mpii.cpu().numpy()

                batch_len = theta_mpii_batch.shape[0]
                for i in range(batch_len):
                    theta_mpii = theta_mpii_batch[i]
                    theta_mano = mpii_to_mano_torch(theta_mpii)

                    xyz_FK = render.hand_mesh.set_abs_xyz_torch(theta_mano) * 10.0

                    # xyz_FK : mano order
                    xyz_ori = xyz[i].type(torch.FloatTensor).cuda()
                    xyz_ori = mpii_to_mano_torch(xyz_ori)

                    loss_FK += IKNet.ForwardKinematic_loss(xyz_FK, xyz_ori)

                    # xyz_FK_np = xyz_FK.detach().cpu().numpy()
                    # xyz_ori_np = xyz_ori.detach().cpu().numpy()
                    # print("s")

            #t3 = time.time()

            loss_FK.backward(retain_graph=True)
            training_loss_IK += loss_FK.data.cpu().numpy()

            loss.backward()
            optimizer_FCN.step()
            optimizer_IKN.step()

            # if batch > 20:
            #     break
            #t4 = time.time()

            # print("FCN : ", t2 - t1)
            # print("IK : ", t3 - t2)
            # print("backward : ", t4 - t3)

        training_loss_FCN = training_loss_FCN / batch
        training_loss_IK = training_loss_IK / batch
        training_loss = training_loss_FCN + training_loss_IK
        #writer.add_scalars('data/loss', {'train_loss': training_loss}, epoch)
        print("Epoch : {} finished. Training Loss_FCN: {}, Loss_IK: {}.".format(epoch, training_loss_FCN, training_loss_IK))

        # validation and save model
        if epoch % 10 == 0:
            """
            validation_loss_FCN = 0.
            validation_loss_IK = 0.
            with torch.no_grad():
                for batch, data in enumerate(tqdm(testing_dataloader)):
                    image = data[0]
                    if torch.isnan(image).any():
                        raise ValueError('Image error')
                    true = [x.cuda() for x in data[1:-1]]

                    ### FCN ###
                    if flag_extra:
                        print("...")
                    else:
                        pred = model(image.cuda())
                        loss = model.total_loss(pred, true)

                        validation_loss_FCN += loss.data.cpu().numpy()

                    ### IKNet ###
                    if flag_extra:
                        print("")
                    else:
                        hand_points_rel, true_hand_points_rel = IKNet.extract_handkeypoint_batch(pred, true)
                        xyz = torch.tensor(hand_points_rel, requires_grad=True).cuda()
                        _, theta_mpii_batch = IKNet(xyz.float())

                        loss_FK = 0
                        # theta_mpii = theta_mpii.cpu().numpy()
                        batch_len = theta_mpii_batch.shape[0]
                        for i in range(batch_len):
                            theta_mpii = theta_mpii_batch[i]
                            theta_mano = mpii_to_mano_torch(theta_mpii)

                            xyz_FK = render.hand_mesh.set_abs_xyz_torch(theta_mano)
                            # xyz_FK : mano order
                            xyz_ori = xyz[i].type(torch.FloatTensor).cuda()
                            xyz_ori = mpii_to_mano_torch(xyz_ori) * 10.0
                            loss_FK += IKNet.ForwardKinematic_loss(xyz_FK, xyz_ori)

                        validation_loss_IK += loss_FK.data.cpu().numpy()

            validation_loss_FCN = validation_loss_FCN / batch
            validation_loss_IK = validation_loss_IK / batch
            validation_loss = validation_loss_FCN + validation_loss_IK
            #writer.add_scalars('data/loss', {'val_loss : ': validation_loss}, epoch)
            print("Epoch : {} finished. Validation Loss_FCN: {}, Loss_IK".format(epoch, validation_loss_FCN, validation_loss_IK))
            """
            torch.save(model.state_dict(), model_FCN_name)
            torch.save(IKNet.iknet.state_dict(), model_IKNet_name)
            print("model saved")

