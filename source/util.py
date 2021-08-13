import sys
import random
import torch
import numpy as np


class extrapolation():
    def __init__(self, batch_len, vis_threshold):
        self.prev_idx = '-1'
        self.prev_batch_idx = '-1'

        self.prev_gt_3d = torch.zeros([2, 21, 4], dtype=torch.float32)
        #self.prev_gt_3d = torch.zeros([2, 63, 5, 13, 13], dtype=torch.float32)
        #self.prev_gt_vis = torch.zeros([2, 21], dtype=torch.float32)
        self.batch_len = batch_len
        self.vis_threshold = vis_threshold

    """
    def grid_extrapolate(self, subject, curr_gt, vis_gt):
        # curr_gt (batch, 63, 5, 13, 13)
        # self.prev_gt_3d (2, 63, 5, 13, 13)

        batch_len = curr_gt.shape[0]
        batch_len = len(subject)
        flag_pass = False

        for i in range(batch_len):
            curr_idx = subject[i]
            prev_vis = self.stacked_gt[i+1, :, -1]
            #prev_vis = prev_vis.cpu().numpy()
            vis_mask = prev_vis < self.vis_threshold

        self.prev_idx = curr_idx
        self.prev_batch_idx = self.prev_idx

        self.prev_gt_3d = curr_gt[-2:, :, :, :, :]
        self.prev_gt_vis = vis_gt[-2:, :]

        return extra
    """

    def grid_to_3d(self, curr_gt, hand_mask_list, vis_prev):
        batch_len = curr_gt.shape[0]

        for i in range(batch_len):
            hand_mask = hand_mask_list[i].unsqueeze(0)
            true_hand_cell = np.unravel_index(hand_mask.argmax(), hand_mask.shape)
            z, v, u = true_hand_cell[1:]
            dels = curr_gt[i, :, z, v, u].reshape(21, 3)
            vis = vis_prev[i, :].unsqueeze(1)

            dels = torch.cat([dels, vis], dim=1)
            if i == 0:
                self.curr_gt_3d = dels.unsqueeze(0)
            else:
                self.curr_gt_3d = torch.cat([self.curr_gt_3d, dels.unsqueeze(0)], dim=0)

        self.stacked_gt = torch.cat([self.prev_gt_3d, self.curr_gt_3d], dim=0).cuda()
        self.prev_gt_3d = self.curr_gt_3d[-2:, ]

        return self.stacked_gt

    def extrapolate(self, subject):
        batch_len = len(subject)
        flag_pass = False

        for i in range(batch_len):
            curr_idx = subject[i]
            prev_vis = self.stacked_gt[i+1, :, -1]
            #prev_vis = prev_vis.cpu().numpy()
            vis_mask = prev_vis < self.vis_threshold

            if flag_pass:
                flag_pass = False
                self.prev_idx = curr_idx
                continue

            if i == 0:
                if curr_idx != self.prev_batch_idx:
                    if batch_len == 1:
                        extra = torch.zeros([1, 21, 3], dtype=torch.float32).cuda()
                    else:
                        extra = torch.zeros([2, 21, 3], dtype=torch.float32).cuda()
                    flag_pass = True
                else:
                    extra = 2 * self.stacked_gt[1, :, :-1] - self.stacked_gt[0, :, :-1]
                    # if the value of same idx in prev_vis is 0 (joint is occluded)
                    # then the joint has relative position of t-1
                    extra[vis_mask, :] = self.stacked_gt[1, vis_mask, :-1] - self.stacked_gt[1, 0, :-1]
                    extra = extra.unsqueeze(0)

            else:
                if curr_idx != self.prev_idx:
                    if i != (batch_len - 1):
                        ex = torch.zeros([2, 21, 3], dtype=torch.float32).cuda()
                        flag_pass = True
                    else:
                        ex = torch.zeros([1, 21, 3], dtype=torch.float32).cuda()
                    extra = torch.cat([extra, ex], dim=0)

                else:
                    # i == t-2, i+1 == t-1
                    ex = 2 * self.stacked_gt[i + 1, :, :-1] - self.stacked_gt[i, :, :-1]
                    ex[vis_mask, :] = self.stacked_gt[i + 1, vis_mask, :-1] - self.stacked_gt[i + 1, 0, :-1]
                    extra = torch.cat([extra, ex.unsqueeze(0)], dim=0)

            self.prev_idx = curr_idx

            self.prev_batch_idx = self.prev_idx

        return extra

    def grid_to_3d_test(self, curr_gt, hand_mask_list, vis_prev):
        batch_len = curr_gt.shape[0]

        for i in range(batch_len):
            hand_mask = hand_mask_list[i].unsqueeze(0)
            true_hand_cell = np.unravel_index(hand_mask.argmax(), hand_mask.shape)
            z, v, u = true_hand_cell[1:]
            dels = curr_gt[i, :, z, v, u].reshape(21, 3)
            vis = vis_prev[i, :].unsqueeze(1)

            dels = torch.cat([dels, vis], dim=1)
            if i == 0:
                self.curr_gt_3d = dels.unsqueeze(0)
            else:
                self.curr_gt_3d = torch.cat([self.curr_gt_3d, dels.unsqueeze(0)], dim=0)

        self.stacked_gt = torch.cat([self.prev_gt_3d, self.curr_gt_3d], dim=0).cuda()
        self.prev_gt_3d = self.curr_gt_3d[-2:, ]

        return self.stacked_gt

    def extrapolate_test(self, pred_prev):
        # pred_prev : torch(2, 21, 4)
        prev_vis = pred_prev[1, :, -1]
        # prev_vis = prev_vis.cpu().numpy()
        vis_mask = prev_vis < self.vis_threshold

        extra = 2 * pred_prev[1, :, :-1] - pred_prev[0, :, :-1]
        # if the value of same idx in prev_vis is 0 (joint is occluded)
        # then the joint has relative position of t-1
        extra[vis_mask, :] = pred_prev[1, vis_mask, :-1] - pred_prev[1, 0, :-1]
        extra = extra.unsqueeze(0)

        return extra
