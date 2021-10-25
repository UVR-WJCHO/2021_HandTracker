import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append("./IKNet")

import argparse
import time
import random
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from einops import rearrange
from open3d import io as io

from cfg import parameters
from net import *
from dataset import HO3D_v2_Dataset, FHAD_Dataset, FreiHAND_Dataset
from vis_utils.vis_utils import *

import IKNet.config as config
from IKNet.utils import *
from IKNet.model.hand_mesh import minimal_hand
from IKNet.render import o3d_render
from IKNet.kinematics import *
from IKNet.hand_mesh import HandMesh
from IKNet.kinematics import mpii_to_mano
from IKNet.capture import OpenCVCapture


def _log_parameters(args):
    log_name = '../models/log_' + args['model'] + '.txt'
    with open(log_name, mode='wt', encoding='utf-8') as f:
        for arg in args:
            f.write(arg + ': ' + str(args[arg]) + '\n')

def _log_loss(args, log_name=None, init=False):
    if init:
        log_name = '../models/log_' + args['model'] + '_loss.txt'
        with open(log_name, mode='wt', encoding='utf-8') as f:
            f.write('[Loss log]\n')
        return log_name
    else:
        with open(log_name, mode='at', encoding='utf-8') as f:
            f.writelines(args)


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-model", required=False, type=str, default='FCN_1025_HO3Donly')
    ap.add_argument("-load_model", type=str, default=None)
    ap.add_argument("-continue_train", required=False, type=int, default=0, help="continue train on epoch ~")
    ap.add_argument("-epoch", required=False, type=int, default=100)
    ap.add_argument("-lr", required=False, type=float, default=0.0001)
    ap.add_argument("-num_worker", required=False, type=int, default=4)

    ap.add_argument("-extra", required=False, type=bool, default=False, help="activate extrapolation")   # action='store_true'
    ap.add_argument("-augment", required=False, type=bool, default=True, help="activate augmentation")

    ap.add_argument("-res34", required=False, type=bool, default=False, help="use res34 backbone")
    ap.add_argument("-lowerdim", required=False, type=bool, default=True, help="concatenate extra feature on lower part of network")

    ap.add_argument("-dataset", required=False, choices=['ho3d', 'wfhad', 'wfrei', 'all'], default='ho3d', help="choose dataset option to train")
    args = vars(ap.parse_args())

    _log_parameters(args)
    log_loss_name = _log_loss(args, init=True)
    model_name = '../models/' + args['model']

    ############## Load dataset and set dataloader ##############
    # training_dataset_HO3D = HO3D_v2_Dataset_before(mode='train', cfg='train', loadit=True, augment=True, extra=args['extra'])
    # validating_dataset_HO3D = HO3D_v2_Dataset_before(mode='train', cfg='test', loadit=True, extra=args['extra'])

    training_dataset_HO3D = HO3D_v2_Dataset(mode='train', cfg='train', loadit=True, augment=True, extra=args['extra'])
    validating_dataset_HO3D = HO3D_v2_Dataset(mode='train', cfg='test', loadit=True, extra=args['extra'])

    training_dataloader = torch.utils.data.DataLoader(training_dataset_HO3D, batch_size=parameters.batch_size,
                                                      shuffle=True,
                                                      num_workers=args['num_worker'], pin_memory=True)
    validating_dataloader = torch.utils.data.DataLoader(validating_dataset_HO3D, batch_size=parameters.batch_size,
                                                        shuffle=False,
                                                        num_workers=args['num_worker'], pin_memory=True)

    # FHAD dataset
    if args['dataset'] in ['wfhad', 'all']:
        # currupted rgb img
        training_dataset_FHAD = FHAD_Dataset(mode='train', cfg='train', loadit=True, augment=True, extra=args['extra'])
        training_dataloader_FHAD = torch.utils.data.DataLoader(training_dataset_FHAD, batch_size=parameters.batch_size,
                                                               shuffle=True,
                                                               num_workers=args['num_worker'], pin_memory=True)
    # FreiHAND dataset
    if args['dataset'] in ['wfrei', 'all']:
        # only rgb, only handpose ~ no visibility & extrapolated feature
        training_dataset_FreiHAND = FreiHAND_Dataset(mode='train', loadit=True, augment=False)

        training_dataloader_FreiHAND = torch.utils.data.DataLoader(training_dataset_FreiHAND, batch_size=parameters.batch_size,
                                                               shuffle=True,
                                                               num_workers=args['num_worker'], pin_memory=True)

    ############## Set model and training parameters ##############
    device = torch.device('cuda:0')
    if args['res34']:
        model = UnifiedNet_res34()
        if args['lowerdim']:
            model = UnifiedNet_res34_lowconcat()
    else:
        model = UnifiedNet_res18()
        if args['lowerdim']:
            model = UnifiedNet_res18_lowconcat()
    net_utils = Network_utils()
    # model.load_state_dict(torch.load('../models/unified_net_addextra.pth', map_location=str(device)), strict=False)

    param_num = sum(p.numel() for p in model.parameters() if p.requires_grad)  #print("param num : ", param_num)
    assert (torch.cuda.is_available())

    ### if continuing training ###
    if args['continue_train'] != 0:
        if args['load_model'] is None:
            args['load_model'] = args['model']
        model_load_name = '../models/' + args['load_model'] + '_epoch_' + str(args['continue_train']) + '.pth'
        state_dict = torch.load(model_load_name, map_location=str(device))
        model.load_state_dict(state_dict, strict=False)
        print("load success")
    model.cuda()

    lr_FCN = args['lr']
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_FCN)
    best_loss = float('inf')
    start_epoch = args['continue_train']
    end_epoch = args['epoch']

    ############## Main loop ##############
    model.train()
    for epoch in range(start_epoch, end_epoch):
        ### update learning rate ###
        if epoch == 60 or epoch == 120:
            lr_FCN *= 0.1
            optimizer = torch.optim.Adam(model.parameters(), lr=lr_FCN)

        ### start training HO3D dataset ###
        print("...training HO3D")
        model.update_parameter(img_width=640., img_height=480.)
        model.check_status()

        training_loss = 0.
        t0 = time.time()
        for batch, data in enumerate(tqdm(training_dataloader)):
            optimizer.zero_grad()
            image = data[0]
            # if torch.isnan(image).any():
            #     raise ValueError('Image error')
            true = [x.cuda() for x in data[1:-2]]

            if args['extra']:
                extra = data[-1]
                pred = model(image.cuda(), extra.cuda())
            else:
                pred = model(image.cuda())

            loss = model.total_loss(pred, true)
            loss.backward()
            optimizer.step()
            training_loss += loss.detach().cpu().numpy()
        training_loss = training_loss / batch
        log_loss = "Epoch : {}. HO3D Training loss: {}.\n".format(epoch, training_loss)
        _log_loss(log_loss, log_name=log_loss_name)
        print(log_loss)

        if args['dataset'] in ['wfhad', 'all']:
            ### start training FHAD dataset ###
            print("...training FHAD")
            model.update_parameter(img_width=1920., img_height=1080.)
            model.check_status()

            training_loss = 0.
            assert training_loss == 0, 'loss initialization error'
            for batch, data in enumerate(tqdm(training_dataloader_FHAD)):
                # t1 = time.time()
                optimizer.zero_grad()

                image = data[0]
                # if torch.isnan(image).any():
                #     raise ValueError('Image error')
                true = [x.cuda() for x in data[1:-1]]

                if args['extra']:
                    extra = data[-1]
                    pred = model(image.cuda(), extra.cuda())
                else:
                    pred = model(image.cuda())

                loss = model.total_loss_FHAD(pred, true)
                loss.backward()
                optimizer.step()
                training_loss += loss.data.detach().cpu().numpy()
            training_loss = training_loss / batch
            log_loss = "Epoch : {}. FHAD Training loss: {}.\n".format(epoch, training_loss)
            _log_loss(log_loss, log_name=log_loss_name)
            print(log_loss)

        if args['dataset'] in ['wfrei', 'all']:
            ### start training FreiHAND dataset ###
            print("...training FreiHAND")
            model.update_parameter(img_width=224., img_height=224.)
            model.check_status()

            training_loss = 0.
            assert training_loss == 0, 'loss initialization error'
            for batch, data in enumerate(tqdm(training_dataloader_FreiHAND)):
                # t1 = time.time()
                optimizer.zero_grad()

                image = data[0]
                # if torch.isnan(image).any():
                #     raise ValueError('Image error')
                true = [x.cuda() for x in data[1:]]

                pred = model(image.cuda())

                loss = model.total_loss_FreiHAND(pred, true)
                loss.backward()
                optimizer.step()
                training_loss += loss.data.detach().cpu().numpy()
            training_loss = training_loss / batch
            log_loss = "Epoch : {}. FreiHAND Training loss: {}.\n".format(epoch, training_loss)
            _log_loss(log_loss, log_name=log_loss_name)
            print(log_loss)

        ### validate on HO3D dataset and save model ###
        if epoch != 0 and epoch % 5 == 0:
            validation_loss = 0.

            with torch.no_grad():
                for batch, data in enumerate(tqdm(validating_dataloader)):
                    image = data[0]
                    true = [x.cuda() for x in data[1:-2]]
                    if args['extra']:
                        extra = data[-1]
                        pred = model(image.cuda(), extra.cuda())
                    else:
                        pred = model(image.cuda())
                    loss = model.total_loss(pred, true)

                    validation_loss += loss.data.detach().cpu().numpy()

            validation_loss = validation_loss / batch
            log_loss = "Epoch : {} finished. Validation loss on HO3D: {}.\n".format(epoch, validation_loss)
            _log_loss(log_loss, log_name=log_loss_name)
            print(log_loss)

            save_model_FCN_name = model_name + '_epoch_' + str(epoch) + '.pth'
            torch.save(model.state_dict(), save_model_FCN_name)
            print("model saved")

    save_model_FCN_name = model_name + '.pth'
    torch.save(model.state_dict(), save_model_FCN_name)
    print("model saved")
    print("training finished")