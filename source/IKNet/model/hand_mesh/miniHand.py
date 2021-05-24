'''
detnet based on PyTorch
version: 1.0
author: lingteng qiu 
email: qiulingteng@link.cuhk.edu.cn
'''
import torch
import sys
sys.path.append("./")

from torch import nn, einsum
from einops import rearrange,repeat
import torch.nn.functional as F
import torchvision
import numpy as np
import pickle
from IKNet.model.hand_mesh.kinematics import *
from IKNet.model.detnet import detnet
from IKNet.model.iknet import iknet


class minimal_hand(nn.Module):
    def __init__(self,mano_path,iknet_path=None):
        super().__init__()
        self.para_init(mano_path)
        #self.detnet = detnet(stacks = 1)
        self.iknet = iknet(inc = 84*3,depth = 6, width = 1024)

        # initialize all param
        #torch.nn.init.xavier_uniform(self.iknet.weight)

        # load model
        self.model_init(iknet_path)
        # if extra == True, self.model_init_ext(iknet_path)
        self.setup_losses()

    def setup_losses(self):
        self.ForwardKinematic_loss = nn.MSELoss()

    def para_init(self,mano_path):
        '''
        para_init
        mpii_ref_delta means the longth of bone between children joints and parent joints note that root is equal to itself.
        '''
        self.__ik = 0.09473151311686484
        mano_ref_xyz = self.__load_pkl(mano_path)['joints']
        mpii_ref_xyz = mano_to_mpii(mano_ref_xyz) / self.__ik
        mpii_ref_xyz -= mpii_ref_xyz[9:10]
        mpii_ref_delta, mpii_ref_length = xyz_to_delta(mpii_ref_xyz, MPIIHandJoints)
        mpii_ref_delta = mpii_ref_delta * mpii_ref_length

        self.__mpii_ref_xyz = torch.from_numpy(mpii_ref_xyz).float()
        self.__mpii_ref_delta = torch.from_numpy(mpii_ref_delta).float()

        self.camera_intrinsics = np.array([[1395.749023, 0, 935.732544],
                                           [0, 1395.749268, 540.681030],
                                           [0, 0, 1]])

    def forward(self, xyz_batch):
        device = xyz_batch.device

        # 1 batch (testing)
        if xyz_batch.shape[0] == 21:
            xyz = xyz_batch
            delta, length = xyz_to_delta_tensor(xyz, MPIIHandJoints, device=device)
            delta *= length
            pack = torch.cat([xyz, delta, self.mpii_ref_xyz.to(device), self.mpii_ref_delta.to(device)],
                             dim=0).unsqueeze(0)
        # multiple batch
        else:
            xyz = xyz_batch[0]
            # this 11-12 delta have some mistake, need to check why the different is so high.
            delta, length = xyz_to_delta_tensor(xyz, MPIIHandJoints, device=device)
            delta *= length
            pack = torch.cat([xyz, delta, self.mpii_ref_xyz.to(device), self.mpii_ref_delta.to(device)], dim=0).unsqueeze(0)

            for i in range(xyz_batch.shape[0] - 1):
                xyz = xyz_batch[i+1]
                delta, length = xyz_to_delta_tensor(xyz, MPIIHandJoints, device=device)
                delta *= length
                pack_stack = torch.cat([xyz, delta, self.mpii_ref_xyz.to(device), self.mpii_ref_delta.to(device)],
                                 dim=0).unsqueeze(0)
                pack = torch.cat([pack, pack_stack], dim=0)

        # pack : (batch, 84 ,3)
        theta, _ = self.iknet(pack)

        return xyz, theta #[0]


    def forward_withoutbatch_test(self, xyz):
        # b c h w == 128
        #uv, xyz = self.detnet(x)
        device = xyz.device

        #this 11-12 delta have some mistake, need to check why the different is so high.
        delta, length = xyz_to_delta(xyz, MPIIHandJoints,device=device)

        delta *= length
        pack = torch.cat([xyz, delta, self.mpii_ref_xyz.to(device), self.mpii_ref_delta.to(device)],dim=0).unsqueeze(0)

        # pack : (batch, 84 ,3)
        theta, _ = self.iknet(pack)
        
        return xyz, theta[0]    # batch가 1인 것을 가정한 코드..

    def forward_backup(self, x):
        # b c h w == 128
        uv, xyz = self.detnet(x)
        device = xyz.device

        # this 11-12 delta have some mistake, need to check why the different is so high.
        delta, length = xyz_to_delta_tensor(xyz, MPIIHandJoints, device=device)

        delta *= length
        pack = torch.cat([xyz, delta, self.mpii_ref_xyz.to(device), self.mpii_ref_delta.to(device)], dim=0).unsqueeze(0)

        theta = self.iknet(pack)[0]

        return xyz, theta[0]

    def extract_handkeypoint(self, pred, true):
        pred_hand_pose, _, _, _, _, _ = [p.data.cpu().numpy() for p in pred]
        true_hand_pose, _, hand_mask, _, _, _ = [t.data.cpu().numpy() for t in true]

        pred_hand_pose = np.squeeze(pred_hand_pose)
        true_hand_pose = np.squeeze(true_hand_pose)
        hand_mask = np.squeeze(hand_mask)

        ## change numpy to pytorch (cpu > gpu)
        true_hand_cell = np.unravel_index(hand_mask.argmax(), hand_mask.shape)
        z, v, u = true_hand_cell
        dels = pred_hand_pose[:, z, v, u].reshape(21, 3)
        del_u, del_v, del_z = dels[:, 0], dels[:, 1], dels[:, 2]
        hand_points, xy_points = self.target_to_control(del_u, del_v, del_z, (u, v, z))

        dels = true_hand_pose[:, z, v, u].reshape(21, 3)
        del_u, del_v, del_z = dels[:, 0], dels[:, 1], dels[:, 2]
        true_hand_points, _ = self.target_to_control(del_u, del_v, del_z, (u, v, z))

        root = true_hand_points[9, :]
        true_hand_points_rel = (root - true_hand_points) / 100.

        root = hand_points[9, :]
        hand_points_rel = (root - hand_points) / 100.

        return hand_points_rel, true_hand_points_rel, root, hand_points, true_hand_points
        
    def extract_handkeypoint_batch(self, pred, true):
        pred_hand_pose_b, _, _, _, _, _ = [p.data.cpu().numpy() for p in pred]
        true_hand_pose_b, _, hand_mask_b, _, _, _ = [t.data.cpu().numpy() for t in true]

        batch_len = pred_hand_pose_b.shape[0]

        # craete output array (batch, 21, 3)
        hand_points_rel_b = np.zeros((batch_len, 21, 3))
        true_hand_points_rel_b = np.zeros((batch_len, 21, 3))

        for i in range(batch_len):
            pred_hand_pose = pred_hand_pose_b[i]
            true_hand_pose = true_hand_pose_b[i]
            hand_mask = hand_mask_b[i]

            ## change numpy to pytorch (cpu > gpu)
            true_hand_cell = np.unravel_index(hand_mask.argmax(), hand_mask.shape)
            z, v, u = true_hand_cell
            dels = pred_hand_pose[:, z, v, u].reshape(21, 3)
            del_u, del_v, del_z = dels[:, 0], dels[:, 1], dels[:, 2]
            hand_points, xy_points = self.target_to_control(del_u, del_v, del_z, (u, v, z))

            dels = true_hand_pose[:, z, v, u].reshape(21, 3)
            del_u, del_v, del_z = dels[:, 0], dels[:, 1], dels[:, 2]
            true_hand_points, _ = self.target_to_control(del_u, del_v, del_z, (u, v, z))

            root = true_hand_points[9, :]
            true_hand_points_rel = (root - true_hand_points) / 100.

            root = hand_points[9, :]
            hand_points_rel = (root - hand_points) / 100.

            hand_points_rel_b[i] = hand_points_rel
            true_hand_points_rel_b[i] = true_hand_points_rel

        return hand_points_rel_b, true_hand_points_rel_b

    def target_to_control(self, del_u, del_v, del_z, cell):
        u, v, z = cell

        w_u = del_u + u
        w_v = del_v + v
        w_z = del_z + z

        w_u, w_v, w_z = self.upsample_points((w_u, w_v), w_z)
        points = np.vstack((w_u * 32, w_v * 32, np.ones_like(w_u)))
        y_hat = w_z * 15 * np.linalg.inv(self.camera_intrinsics).dot(points)

        return y_hat.T, points

    def upsample_points(self, points, depth):

        downsample_ratio_x = 1920 / 416.
        downsample_ratio_y = 1080 / 416.

        u = points[0] * downsample_ratio_x
        v = points[1] * downsample_ratio_y
        z = depth * 10.  # converting to millimeters

        return u, v, z

    @property
    def ik_unit_length(self):
        return self.__ik
    @property
    def mpii_ref_xyz(self):
        return self.__mpii_ref_xyz
    @property
    def mpii_ref_delta(self):
        return self.__mpii_ref_delta

    def model_init_ext(self, iknet_saved):
        if iknet == None:
            raise NotImplementedError
        pretrained_dict = iknet_saved.state_dict()
        new_model_dict = self.iknet.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in new_model_dict}
        new_model_dict.update(pretrained_dict)
        self.iknet.load_state_dict(new_model_dict)


    def model_init(self, iknet):
        if iknet == None:
            raise NotImplementedError
        self.iknet.load_state_dict(torch.load(iknet))

    def model_init_backup(self,detnet,iknet):
        if detnet == None:
            raise NotImplementedError
        if iknet == None:
            raise NotImplementedError
        self.detnet.load_state_dict(torch.load(detnet))
        self.iknet.load_state_dict(torch.load(iknet))
        
    def __load_pkl(self,path):
        """
        Load pickle data.
        Parameter
        ---------
        path: Path to pickle file.
        Return
        ------
        Data in pickle file.
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data

if __name__ == '__main__':
    hand_machine = minimal_hand('../weights/minimal_hand/model/hand_mesh/hand_mesh_model.pkl','./weights/detnet.pth','./weights/iknet.pth')
    hand_machine.eval()
    inp  = np.load("./input.npy")
    output = np.load('./output.npy')
    output = torch.from_numpy(output)
    output = rearrange(output,'b h w c -> b c h w')
    inp = torch.from_numpy(inp)
    inp = rearrange(inp,'b h w c -> b c h w')  

    x = np.load("iknet_inputs.npy")
    x = torch.from_numpy(x).float()

    gt = np.load("theta.npy")
    gt = torch.from_numpy(gt).float()
    hand_machine.cuda()
    hand_machine(inp.cuda())
    