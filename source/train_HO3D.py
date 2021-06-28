import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append("./IKNet")

import time
import random
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from einops import rearrange
from open3d import io as io

from cfg import parameters
from net import UnifiedNetwork, UnifiedNetwork_update
from dataset import HO3D_v2_Dataset
from util import extrapolation

from vis_utils.vis_utils import *
import IKNet.config as config
from IKNet.utils import *
from IKNet.model.hand_mesh import minimal_hand
from IKNet.render import o3d_render
from IKNet.kinematics import *
from IKNet.hand_mesh import HandMesh
from IKNet.kinematics import mpii_to_mano
from IKNet.capture import OpenCVCapture


if __name__ == '__main__':

    flag_extra = False
    flag_suffle = True

    continue_train = False
    load_epoch = 0

    model_FCN_name = '../models/FCN_HO3D_trial_1.pth'
    HAND_MESH_MODEL_PATH = './IKNet/IKmodel/hand_mesh/hand_mesh_model.pkl'

    training_dataset_HO3D = HO3D_v2_Dataset(mode='train', loadit=True, shuffle=flag_suffle)
    testing_dataset_HO3D = HO3D_v2_Dataset(mode='test', loadit=True, shuffle=flag_suffle)

    training_dataloader = torch.utils.data.DataLoader(training_dataset_HO3D, batch_size=parameters.batch_size, shuffle=False,
                                                      num_workers=4)
    testing_dataloader = torch.utils.data.DataLoader(testing_dataset_HO3D, batch_size=parameters.batch_size, shuffle=False,
                                                     num_workers=4)

    device = torch.device('cuda:0')

    extp_module = extrapolation()
    model = UnifiedNetwork_update()
    model.load_state_dict(torch.load('../models/unified_net_addextra.pth', map_location=str(device)), strict=False)

    assert (torch.cuda.is_available())

    if continue_train:
        state_dict = torch.load(model_FCN_name, map_location=str(device))
        model.load_state_dict(state_dict, strict=False)
        print("load success")

    model.cuda()

    lr_FCN = 0.0001
    optimizer_FCN = torch.optim.Adam(model.parameters(), lr=lr_FCN)

    best_loss = float('inf')

    epoch_range = parameters.epochs
    if continue_train:
        epoch_range -= load_epoch

    for epoch in range(epoch_range):
        # train
        model.train()
        training_loss_FCN = 0.
        training_loss_IK = 0.

        for batch, data in enumerate(tqdm(training_dataloader)):
            #t1 = time.time()
            optimizer_FCN.zero_grad()

            image = data[0]
            if torch.isnan(image).any():
                raise ValueError('Image error')

            true = [x.cuda() for x in data[1:]]   # added sequence index in dataset

            # vis = data[-1].cuda()

            ############################ FCN ############################
            if flag_extra:
                seq_idx = data[-1]  # tupple, string
                curr_gt = data[1]
                hand_mask_list = data[3]
                batch_len = hand_mask_list.shape[0]
                stacked_gt = extp_module.grid_to_3d(curr_gt, hand_mask_list, batch_len).cuda()
                # stacked_gt : torch, torch.Size([batchsize+2, 22, 3]), gpu

                # At initial or if training dataset's sequence # changed, apply zero extra_keypoint
                # if not, extrapolate keypoint from previous pred
                extra = extp_module.extrapolate(batch_len, seq_idx)
                # extra : (batch_size, 22, 3)

                pred = model(image.cuda(), extra)
                loss = model.total_loss(pred, true)
                training_loss += loss.data.cpu().numpy()

            else:
                pred = model(image.cuda())
                loss = model.total_loss(pred, true)

                training_loss_FCN += loss.data.cpu().numpy()

            loss.backward()
            optimizer_FCN.step()

        training_loss_FCN = training_loss_FCN / batch
        training_loss = training_loss_FCN
        print("Epoch : {} finished. Training Loss_FCN: {}.".format(epoch, training_loss_FCN))

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
            # torch.save(IKNet.iknet.state_dict(), model_IKNet_name)
            print("model saved")

    print("training finished")