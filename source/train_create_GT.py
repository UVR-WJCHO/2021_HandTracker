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
from dataset import HO3D_v2_Dataset_update



if __name__ == '__main__':

    activate_extra = True
    continue_train = False
    load_epoch = 0



    model_FCN_name = '../models/FCN_HO3D_0831_wVis_extra.pth'

    load_model_FCN_name = '../models/FCN_HO3D_0831_wVis_extra_20epoch.pth'
    HAND_MESH_MODEL_PATH = './IKNet/IKmodel/hand_mesh/hand_mesh_model.pkl'
    # dataset pkl are aligned
    # To shuffle the dataset w.r.t subject : set shuffle_seq=True
    # To shuffle the dataset totally : set shuffle=True in DataLoader
    training_dataset_HO3D = HO3D_v2_Dataset_update(mode='train', cfg='train', loadit=True)
    validating_dataset_HO3D = HO3D_v2_Dataset_update(mode='train', cfg='test', loadit=True)

    train_len = training_dataset_HO3D.sample_len
    valid_len = validating_dataset_HO3D.sample_len

    # trainGT = dict()
    # for batch, data in enumerate(tqdm(training_dataloader)):
    #     temp = data[1:]
    #     trainGT[batch] = temp
    #
    # np.save("trainGT.npy", trainGT)
    # print("trainGT saved")

    trainGT = dict()
    print("valid_len : ", valid_len)
    for i in range(valid_len):
        if i % 10 == 0:
            print('itr : ', i)

        image, true_hand_pose, hand_mask, true_object_pose, object_mask, param_vis, vis, extra_hand_pose, flag_set = validating_dataset_HO3D.preprocess(i)

"""
        if i == 0:
            true_hand_pose_stck = np.expand_dims(true_hand_pose.numpy(), axis=0)
            hand_mask_stck = np.expand_dims(hand_mask.numpy(), axis=0)
            true_object_pose_stck = np.expand_dims(true_object_pose.numpy(), axis=0)
            object_mask_stck = np.expand_dims(object_mask.numpy(), axis=0)
            param_vis_stck = np.expand_dims(param_vis.numpy(), axis=0)
            vis_stck = np.expand_dims(vis, axis=0)
            extra_hand_pose_stck = np.expand_dims(extra_hand_pose.numpy(), axis=0)
            flag_set_stck = np.expand_dims(np.array(flag_set), axis=0)
        else:
            true_hand_pose_stck = np.vstack([true_hand_pose_stck, np.expand_dims(true_hand_pose.numpy(), axis=0)])
            hand_mask_stck = np.vstack([hand_mask_stck, np.expand_dims(hand_mask.numpy(), axis=0)])
            true_object_pose_stck = np.vstack([true_object_pose_stck, np.expand_dims(true_object_pose.numpy(), axis=0)])
            object_mask_stck = np.vstack([object_mask_stck,  np.expand_dims(object_mask.numpy(), axis=0)])
            param_vis_stck = np.vstack([param_vis_stck, np.expand_dims(param_vis.numpy(), axis=0)])
            vis_stck = np.vstack([vis_stck, np.expand_dims(vis, axis=0)])
            extra_hand_pose_stck = np.vstack([extra_hand_pose_stck, np.expand_dims(extra_hand_pose.numpy(), axis=0)])
            flag_set_stck = np.vstack([flag_set_stck, np.expand_dims(np.array(flag_set), axis=0)])

    np.save("./preproc_data/true_hand_pose.npy", true_hand_pose_stck)
    np.save("./preproc_data/hand_mask_stck.npy", hand_mask_stck)
    np.save("./preproc_data/true_object_pose_stck.npy", true_object_pose_stck)
    np.save("./preproc_data/object_mask_stck.npy", object_mask_stck)
    np.save("./preproc_data/param_vis_stck.npy", param_vis_stck)
    np.save("./preproc_data/vis_stck.npy", vis_stck)
    np.save("./preproc_data/extra_hand_pose_stck.npy", extra_hand_pose_stck)
    np.save("./preproc_data/flag_set_stck.npy", flag_set_stck)

    
    print("validGT saved")
"""