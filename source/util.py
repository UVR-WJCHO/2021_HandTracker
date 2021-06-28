import sys
import random
import torch
import numpy as np


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
