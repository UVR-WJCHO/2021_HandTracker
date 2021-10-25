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
from net import UnifiedNetwork_v2, UnifiedNetwork_resnet34, UnifiedNetwork_resnet34_lowerdim
from dataset import HO3D_v2_Dataset_update
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
    activate_switch_extra = False
    flag_extra = True
    flag_lowerdim = True
    continue_train = False
    load_epoch = 0

    model_FCN_name = '../models/FCN_1003_extra_low_res34_randvalid.pth'
    if flag_lowerdim:
        model_FCN_name = '../models/FCN_1003_extra_low_res34_randvalid.pth'

    load_model_FCN_name = '../models/FCN_0930_rand_valid.pth'
    HAND_MESH_MODEL_PATH = './IKNet/IKmodel/hand_mesh/hand_mesh_model.pkl'
    # dataset pkl are aligned
    # To shuffle the dataset w.r.t subject : set shuffle_seq=True
    # To shuffle the dataset totally : set shuffle=True in DataLoader
    training_dataset_HO3D = HO3D_v2_Dataset_update(mode='train', cfg='train', loadit=True, augment=True, extra=flag_extra)
    validating_dataset_HO3D = HO3D_v2_Dataset_update(mode='train', cfg='test', loadit=True, extra=flag_extra)

    # initial training dataset is randomized
    training_dataloader = torch.utils.data.DataLoader(training_dataset_HO3D, batch_size=parameters.batch_size, shuffle=True,
                                                      num_workers=4, pin_memory=True)
    validating_dataloader = torch.utils.data.DataLoader(validating_dataset_HO3D, batch_size=parameters.batch_size,
                                                     shuffle=False,
                                                     num_workers=4, pin_memory=True)

    device = torch.device('cuda:0')

    model = UnifiedNetwork_resnet34()
    if flag_lowerdim:
        model = UnifiedNetwork_resnet34_lowerdim()
    #model.load_state_dict(torch.load('../models/unified_net_addextra.pth', map_location=str(device)), strict=False)

    param_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("param num : ", param_num)
    assert (torch.cuda.is_available())

    if continue_train:
        state_dict = torch.load(load_model_FCN_name, map_location=str(device))
        model.load_state_dict(state_dict, strict=False)
        print("load success")
    model.cuda()

    lr_FCN = 0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_FCN)
    best_loss = float('inf')

    epoch_range = 150    #parameters.epochs
    if continue_train:
        epoch_range -= load_epoch

    model.train()
    for epoch in range(epoch_range):
        if epoch == 0:
            epoch = load_epoch
        training_loss = 0.

        if epoch == 60 or epoch == 120:
            lr_FCN *= 0.1
            optimizer = torch.optim.Adam(model.parameters(), lr=lr_FCN)

        for batch, data in enumerate(tqdm(training_dataloader)):
            # t1 = time.time()
            optimizer.zero_grad()

            image = data[0]
            # if torch.isnan(image).any():
            #     raise ValueError('Image error')
            true = [x.cuda() for x in data[1:-2]]

            ############################ FCN ############################
            if flag_extra:
                extra = data[-1]

                pred = model(image.cuda(), extra.cuda())
                loss = model.total_loss(pred, true)

            else:
                pred = model(image.cuda())
                loss = model.total_loss(pred, true)

            loss.backward()
            optimizer.step()
            training_loss += loss.data.detach().cpu().numpy()

        training_loss = training_loss / batch
        print("Epoch : {} finished. Training loss: {}.\n".format(epoch, training_loss))


        # validation and save model
        if epoch != 0 and epoch % 5 == 0:
            validation_loss = 0.

            with torch.no_grad():
                for batch, data in enumerate(tqdm(validating_dataloader)):
                    image = data[0]
                    true = [x.cuda() for x in data[1:-2]]
                    if flag_extra:
                        extra = data[-1]
                        pred = model(image.cuda(), extra.cuda())
                    else:
                        pred = model(image.cuda())
                    loss = model.total_loss(pred, true)

                    validation_loss += loss.data.detach().cpu().numpy()

            validation_loss = validation_loss / batch
            print("Epoch : {} finished. Validation Loss_FCN: {}.\n".format(epoch, validation_loss))

            save_model_FCN_name = model_FCN_name # model_FCN_name[:-4] + '_epoch' + str(epoch) + '.pth'
            torch.save(model.state_dict(), save_model_FCN_name)
            print("model saved")

    save_model_FCN_name = model_FCN_name  # model_FCN_name[:-4] + '_epoch' + str(epoch) + '.pth'
    torch.save(model.state_dict(), save_model_FCN_name)
    print("model saved")
    print("training finished")