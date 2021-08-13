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
from net import UnifiedNetwork_update, UnifiedNetwork_v2
from dataset import HO3D_v2_Dataset_update
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

    activate_extra = True
    continue_train = False
    load_epoch = 0

    flag_extra = False

    model_FCN_name = '../models/FCN_HO3D_08013_extra.pth'
    HAND_MESH_MODEL_PATH = './IKNet/IKmodel/hand_mesh/hand_mesh_model.pkl'

    # dataset pkl are aligned
    # To shuffle the dataset w.r.t subject : set shuffle_seq=True
    # To shuffle the dataset totally : set shuffle=True in DataLoader
    training_dataset_HO3D = HO3D_v2_Dataset_update(mode='train', cfg='train_align', loadit=True)

    # testing_dataset_HO3D = HO3D_v2_Dataset(mode='test', cfg='test_align', loadit=True, shuffle_seq=False)
    # testing_dataloader = torch.utils.data.DataLoader(testing_dataset_HO3D, batch_size=parameters.batch_size, shuffle=False,
    #                                                  num_workers=4)

    # initial training dataset is randomized
    training_dataloader = torch.utils.data.DataLoader(training_dataset_HO3D, batch_size=parameters.batch_size, shuffle=True,
                                                      num_workers=4)

    device = torch.device('cuda:0')

    # extp_module = extrapolation(parameters.batch_size, vis_threshold=0.5)
    model = UnifiedNetwork_v2()
    #model.load_state_dict(torch.load('../models/unified_net_addextra.pth', map_location=str(device)), strict=False)

    param_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # 590540477
    assert (torch.cuda.is_available())

    if continue_train:
        state_dict = torch.load(model_FCN_name, map_location=str(device))
        model.load_state_dict(state_dict, strict=False)
        print("load success")

    model.cuda()

    lr_FCN = 0.0001
    optimizer_FCN = torch.optim.Adam(model.parameters(), lr=lr_FCN)
    best_loss = float('inf')

    epoch_range = 50    #parameters.epochs
    if continue_train:
        epoch_range -= load_epoch

    model.train()
    for epoch in range(epoch_range):
        if epoch == 0:
            epoch = load_epoch
        training_loss = 0.

        # if epoch != 0 and epoch % 10 == 0:
        #     lr_FCN *= 0.8
        #     optimizer_FCN = torch.optim.Adam(model.parameters(), lr=lr_FCN)

        # after several epoch, reload the dataset and dataloader as sequential but shuffled subject.

        if activate_extra and epoch != 0 and epoch % 10 == 0:
            print("change dataset type")
            if not flag_extra:
                flag_extra = True
            else:
                flag_extra = False

        for batch, data in enumerate(tqdm(training_dataloader)):
            #t1 = time.time()
            optimizer_FCN.zero_grad()

            image = data[0]
            if torch.isnan(image).any():
                raise ValueError('Image error')

            true = [x.cuda() for x in data[1:-2]]

        ############################ FCN ############################
            if flag_extra:
                extra = data[-1]

                pred = model(image.cuda(), extra.cuda())
                loss = model.total_loss(pred, true)
                training_loss += loss.data.cpu().numpy()

            else:
                pred = model(image.cuda())
                loss = model.total_loss(pred, true)

                training_loss += loss.data.cpu().numpy()

            loss.backward()
            optimizer_FCN.step()

        training_loss = training_loss / batch
        print("Epoch : {} finished. Training loss: {}.".format(epoch, training_loss))

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