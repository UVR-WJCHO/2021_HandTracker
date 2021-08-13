import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from cfg import parameters


class UnifiedNetwork_v2(nn.Module):

    def __init__(self):

        super(UnifiedNetwork_v2, self).__init__()

        self.num_hand_control_points = parameters.num_hand_control_points
        self.num_object_control_points = parameters.num_object_control_points
        self.num_objects = parameters.num_objects
        self.depth_discretization = parameters.depth_discretization  # 5

        self.visibility = 21

        model = models.resnet18(pretrained=True)
        self.features_image = nn.Sequential(*list(model.children())[:4])
        self.features_fuse = nn.Sequential(*list(model.children())[4:-2])
        # self.features = nn.Sequential(*list(model.children())[:-2])
        self.features_cat = nn.Conv2d(65, 64, 3, padding=1)

        self.hand_vector_size = 3 * self.num_hand_control_points + 1 + self.visibility  # 63 + 1 + 21, (1 for confidence)
        self.object_vector_size = 3 * self.num_object_control_points + 1 + self.num_objects  # 63 + 1 + 4
        self.target_channel_size = self.depth_discretization * (
                    self.hand_vector_size + self.object_vector_size)  # 5 * 153 = 765

        # prediction layers
        self.conv = nn.Conv2d(512, self.target_channel_size, (3, 3), padding=1, bias=True)

        # losses
        self.setup_losses()

        # extra layers
        self.linear_extra = nn.Linear(84, 10816)
        self.relu_extra = nn.ReLU(True)

        # initialize specific layers
        torch.nn.init.xavier_uniform_(self.features_cat.weight)
        torch.nn.init.xavier_uniform_(self.linear_extra.weight)


    def setup_losses(self):

        self.action_ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.object_ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, x, extra=None):
        batch = x.size()[0]
        height, width = x.size()[2:]

        assert height == width
        assert height % 32 == 0

        target_height, target_width = int(height / 32), int(width / 32)

        if extra is None:

            x = self.features_image(x)  # input x : (batch, 3, 416, 416)

            extra = torch.zeros(batch, 4, 21).cuda()

            extra = extra.view(-1, 4 * 21)
            extra = self.linear_extra(extra)
            extra = self.relu_extra(extra)
            extra = extra.view(batch, -1, 104, 104)
            x = torch.cat([x, extra], dim=1)
            x = self.features_cat(x)

            x = self.features_fuse(x)   # input x : (batch, 64, 104, 104)
            x = self.conv(x)  # input x : (batch, 512, 13, 13)

            # x : (batch, 765, 13, 13)
            x = x.view(-1, self.hand_vector_size + self.object_vector_size, self.depth_discretization, target_height, target_width)

            # now x : (batch, 153, 5, 13, 13)
            pred_v_h = x[:, :self.hand_vector_size, :, :, :]
            pred_v_o = x[:, self.hand_vector_size:, :, :, :]

            # hand specific predictions
            pred_hand_pose = pred_v_h[:, :3 * self.num_hand_control_points, :, :, :]
            pred_hand_pose = pred_hand_pose.view(-1, 21, 3, 5, 13, 13)
            pred_hand_pose_root = torch.sigmoid(pred_hand_pose[:, 0, :, :, :, :].unsqueeze(1))
            pred_hand_pose_without_root = pred_hand_pose[:, 1:, :, :, :, :]
            pred_hand_pose = torch.cat((pred_hand_pose_root, pred_hand_pose_without_root), 1).view(-1, 63, 5, 13, 13)
            #pred_action_prob = pred_v_h[:, 3 * self.num_hand_control_points:-1, :, :, :]
            pred_hand_conf = torch.sigmoid(pred_v_h[:, -1, :, :, :])

            pred_hand_vis = pred_v_h[:, 3 * self.num_hand_control_points:-1, :, :, :]   # (-1, 21, 5, 13, 13)
            pred_hand_vis = torch.sigmoid(pred_hand_vis)

            # object specific predictions
            pred_object_pose = pred_v_o[:, :3 * self.num_object_control_points, :, :, :]
            pred_object_pose = pred_object_pose.view(-1, 21, 3, 5, 13, 13)
            pred_object_pose_root = torch.sigmoid(pred_object_pose[:, 0, :, :, :, :].unsqueeze(1))
            pred_object_pose_without_root = pred_object_pose[:, 1:, :, :, :, :]
            pred_object_pose = torch.cat((pred_object_pose_root, pred_object_pose_without_root), 1).view(-1, 63, 5, 13,
                                                                                                         13)
            #pred_object_prob = pred_v_o[:, 3 * self.num_object_control_points:-1, :, :, :]
            pred_object_conf = torch.sigmoid(pred_v_o[:, -1, :, :, :])

            return pred_hand_pose, pred_hand_conf, pred_object_pose, pred_object_conf, pred_hand_vis

        else:
            # x : (batch, 3, 416, 416)
            x = self.features_image(x)

            # extra : (batch, 63, 5, 13, 13)
            extra = extra.view(-1, 4 * 21)
            extra = self.linear_extra(extra)
            extra = self.relu_extra(extra)
            extra = extra.view(batch, -1, 104, 104)
            x = torch.cat([x, extra], dim=1)
            x = self.features_cat(x)

            # x : (batch, 128, 52, 52)
            x = self.features_fuse(x)

            # x : (batch, 512, 13, 13)
            x = self.conv(x)  # input x : (batch, 512, 13, 13)

            # x : (batch, 765, 13, 13)
            x = x.view(-1, self.hand_vector_size + self.object_vector_size, self.depth_discretization, target_height,
                       target_width)

            # x : (batch, 153, 5, 13, 13)
            pred_v_h = x[:, :self.hand_vector_size, :, :, :]
            pred_v_o = x[:, self.hand_vector_size:, :, :, :]

            # hand specific predictions
            pred_hand_pose = pred_v_h[:, :3 * self.num_hand_control_points, :, :, :]
            pred_hand_pose = pred_hand_pose.view(-1, 21, 3, 5, 13, 13)
            pred_hand_pose_root = torch.sigmoid(pred_hand_pose[:, 0, :, :, :, :].unsqueeze(1))
            pred_hand_pose_without_root = pred_hand_pose[:, 1:, :, :, :, :]
            pred_hand_pose = torch.cat((pred_hand_pose_root, pred_hand_pose_without_root), 1).view(-1, 63, 5, 13, 13)
            #pred_action_prob = pred_v_h[:, 3 * self.num_hand_control_points:-1, :, :, :]
            pred_hand_conf = torch.sigmoid(pred_v_h[:, -1, :, :, :])

            pred_hand_vis = pred_v_h[:, 3 * self.num_hand_control_points:-1, :, :, :]  # (-1, 21, 5, 13, 13)
            pred_hand_vis = torch.sigmoid(pred_hand_vis)

            # object specific predictions
            pred_object_pose = pred_v_o[:, :3 * self.num_object_control_points, :, :, :]
            pred_object_pose = pred_object_pose.view(-1, 21, 3, 5, 13, 13)
            pred_object_pose_root = torch.sigmoid(pred_object_pose[:, 0, :, :, :, :].unsqueeze(1))
            pred_object_pose_without_root = pred_object_pose[:, 1:, :, :, :, :]
            pred_object_pose = torch.cat((pred_object_pose_root, pred_object_pose_without_root), 1).view(-1, 63, 5, 13,
                                                                                                         13)
            #pred_object_prob = pred_v_o[:, 3 * self.num_object_control_points:-1, :, :, :]
            pred_object_conf = torch.sigmoid(pred_v_o[:, -1, :, :, :])

            return pred_hand_pose, pred_hand_conf, pred_object_pose, pred_object_conf, pred_hand_vis

    def total_loss(self, pred, true):

        pred_hand_pose, pred_hand_conf, pred_object_pose, pred_object_conf, pred_hand_vis = pred
        true_hand_pose, hand_mask, true_object_pose, object_mask, true_hand_vis = true

        total_pose_loss = self.pose_loss(pred_hand_pose, true_hand_pose, hand_mask) + self.pose_loss(pred_object_pose,
                                                                                                     true_object_pose,
                                                                                                     object_mask)
        total_pose_loss *= 2.

        total_conf_loss = self.conf_loss(pred_hand_conf, pred_hand_pose, true_hand_pose, hand_mask) + self.conf_loss(
            pred_object_conf, pred_object_pose, true_object_pose, object_mask)
        #total_action_loss = self.action_loss(pred_action_prob, true_action_prob, hand_mask)
        #total_object_loss = self.object_loss(pred_object_prob, true_object_prob, object_mask)

        total_vis_loss = self.vis_loss(pred_hand_vis, true_hand_vis)
        #print("loss : ", total_pose_loss, total_conf_loss, total_vis_loss)
        total_loss = total_pose_loss + total_conf_loss + total_vis_loss#+ total_action_loss + total_object_loss

        return total_loss

    def vis_loss(self, pred, true):

        pred = pred.view(-1, 21, 5, 13, 13)
        true = true.view(-1, 21, 5, 13, 13)
        vis_loss = torch.mean(torch.sum(torch.sum(torch.mul(pred - true, pred - true), dim=[1]), dim=[1, 2, 3]))
        return vis_loss

    def pose_loss(self, pred, true, mask):

        pred = pred.view(-1, 21, 3, 5, 13, 13)
        true = true.view(-1, 21, 3, 5, 13, 13)

        masked_pose_loss = torch.mean(torch.sum(mask * torch.sum(torch.mul(pred - true, pred - true), dim=[1, 2]), dim=[1, 2, 3]))
        return masked_pose_loss

    def conf_loss(self, pred_conf, pred, true, mask):

        pred = pred.view(-1, 21, 3, 5, 13, 13)
        true = true.view(-1, 21, 3, 5, 13, 13)

        pred_pixel_x = pred[:, :, 0, :, :, :].unsqueeze(2) * 32 * (1920. / 416)
        pred_pixel_y = pred[:, :, 1, :, :, :].unsqueeze(2) * 32 * (1080. / 416)
        pred_depth = pred[:, :, 2, :, :, :] * 15 * 10

        true_pixel_x = true[:, :, 0, :, :, :].unsqueeze(2) * 32 * (1920. / 416)
        true_pixel_y = true[:, :, 1, :, :, :].unsqueeze(2) * 32 * (1080. / 416)
        true_depth = true[:, :, 2, :, :, :] * 15 * 10

        pixel_x_distance = torch.mul(pred_pixel_x - true_pixel_x, pred_pixel_x - true_pixel_x)
        pixel_y_distance = torch.mul(pred_pixel_y - true_pixel_y, pred_pixel_y - true_pixel_y)
        pixel_distance = torch.sqrt(pixel_x_distance + pixel_y_distance).squeeze(2)
        depth_distance = torch.sqrt(torch.mul(pred_depth - true_depth, pred_depth - true_depth))

        # threshold
        pixel_distance_mask = (pixel_distance < parameters.pixel_threshold).type(torch.cuda.FloatTensor)
        depth_distance_mask = (depth_distance < parameters.depth_threshold).type(torch.cuda.FloatTensor)

        pixel_distance = pixel_distance / (32 * 416 / 1920.)
        depth_disrance = depth_distance / (15 * 10.)

        pixel_distance = torch.from_numpy(pixel_distance.cpu().data.numpy()).cuda()
        depth_distance = torch.from_numpy(depth_distance.cpu().data.numpy()).cuda()

        pixel_conf = torch.exp(parameters.sharpness * (1 - pixel_distance / parameters.pixel_threshold)) / torch.exp(
            parameters.sharpness * (1 - torch.zeros(pixel_distance.size()).cuda()))
        depth_conf = torch.exp(parameters.sharpness * (1 - depth_distance / parameters.depth_threshold)) / torch.exp(
            parameters.sharpness * (1 - torch.zeros(depth_distance.size()).cuda()))

        pixel_conf = torch.mean(pixel_distance_mask * pixel_conf, dim=1)
        depth_conf = torch.mean(depth_distance_mask * depth_conf, dim=1)

        true_conf = 0.5 * (pixel_conf + depth_conf)
        squared_conf_error = torch.mul(pred_conf - true_conf, pred_conf - true_conf)
        exist_conf_error = torch.mean(torch.sum(mask * squared_conf_error, dim=[1, 2, 3]))

        true_conf = torch.zeros(pred_conf.size()).cuda()
        squared_conf_error = torch.mul(pred_conf - true_conf, pred_conf - true_conf)
        no_exist_conf_error = torch.mean(torch.sum((1 - mask) * squared_conf_error, dim=[1, 2, 3]))

        return 5 * exist_conf_error + 0.1 * no_exist_conf_error

    def action_loss(self, pred, true, mask):
        action_ce_loss = self.action_ce_loss(pred, true)
        return torch.mean(torch.sum(mask * action_ce_loss, dim=[1, 2, 3]))

    def object_loss(self, pred, true, mask):
        object_ce_loss = self.object_ce_loss(pred, true)
        return torch.mean(torch.sum(mask * object_ce_loss, dim=[1, 2, 3]))


class UnifiedNetwork_update(nn.Module):

    def __init__(self):

        super(UnifiedNetwork_update, self).__init__()

        self.num_hand_control_points = parameters.num_hand_control_points
        self.num_object_control_points = parameters.num_object_control_points
        self.num_actions = parameters.num_actions
        self.num_objects = parameters.num_objects
        self.depth_discretization = parameters.depth_discretization  # 5

        model = models.resnet18(pretrained=True)
        self.features_image = nn.Sequential(*list(model.children())[:6])
        self.features_fuse = nn.Sequential(*list(model.children())[6:-2])
        # self.features = nn.Sequential(*list(model.children())[:-2])
        self.features_cat = nn.Conv2d(132, 128, 3, padding=1)

        self.hand_vector_size = 3 * self.num_hand_control_points + 1 + self.num_actions  # 63 + 1 + 4, +1 for confidence
        self.object_vector_size = 3 * self.num_object_control_points + 1 + self.num_objects  # 63 + 1 + 4
        self.target_channel_size = self.depth_discretization * (
                    self.hand_vector_size + self.object_vector_size)  # 5 * 136 = 680

        # prediction layers
        self.conv = nn.Conv2d(512, self.target_channel_size, (3, 3), padding=1, bias=True)
        self.conv_vis = nn.Conv2d(512, self.depth_discretization * 21, (3, 3), padding=1, bias=True)

        # losses
        self.setup_losses()

        # extra layers
        self.linear_extra = nn.Linear(21 * 3, 10816)
        self.relu_extra = nn.ReLU(True)

        # initialize specific layers
        torch.nn.init.xavier_uniform_(self.features_cat.weight)
        torch.nn.init.xavier_uniform_(self.conv_vis.weight)
        torch.nn.init.xavier_uniform_(self.linear_extra.weight)

    def setup_losses(self):

        self.action_ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.object_ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, x, extra=None):
        batch = x.size()[0]
        height, width = x.size()[2:]

        assert height == width
        assert height % 32 == 0

        target_height, target_width = int(height / 32), int(width / 32)

        if extra is None:

            x = self.features_image(x)  # input x : (batch, 3, 416, 416)

            extra = torch.zeros(batch, 21, 3).cuda()
            extra = extra.view(-1, 21 * 3)
            extra = self.linear_extra(extra)
            extra = self.relu_extra(extra)
            extra = extra.view(batch, -1, 52, 52)
            x = torch.cat([x, extra], dim=1)
            x = self.features_cat(x)

            x = self.features_fuse(x)   # input x : (batch, 64, 104, 104)
            x_1 = self.conv(x)  # input x : (batch, 512, 13, 13)
            x_2 = self.conv_vis(x)

            # x : (batch, 680, 13, 13)
            x = x_1.view(-1, self.hand_vector_size + self.object_vector_size, self.depth_discretization, target_height,
                       target_width)
            pred_hand_vis = x_2.view(-1, 21, self.depth_discretization, target_height, target_width)
            pred_hand_vis = torch.sigmoid(pred_hand_vis)

            # now x : (batch, 136, 5, 13, 13)
            pred_v_h = x[:, :self.hand_vector_size, :, :, :]
            pred_v_o = x[:, self.hand_vector_size:, :, :, :]

            # hand specific predictions
            pred_hand_pose = pred_v_h[:, :3 * self.num_hand_control_points, :, :, :]
            pred_hand_pose = pred_hand_pose.view(-1, 21, 3, 5, 13, 13)
            pred_hand_pose_root = torch.sigmoid(pred_hand_pose[:, 0, :, :, :, :].unsqueeze(1))
            pred_hand_pose_without_root = pred_hand_pose[:, 1:, :, :, :, :]
            pred_hand_pose = torch.cat((pred_hand_pose_root, pred_hand_pose_without_root), 1).view(-1, 63, 5, 13, 13)
            #pred_action_prob = pred_v_h[:, 3 * self.num_hand_control_points:-1, :, :, :]
            pred_hand_conf = torch.sigmoid(pred_v_h[:, -1, :, :, :])

            # object specific predictions
            pred_object_pose = pred_v_o[:, :3 * self.num_object_control_points, :, :, :]
            pred_object_pose = pred_object_pose.view(-1, 21, 3, 5, 13, 13)
            pred_object_pose_root = torch.sigmoid(pred_object_pose[:, 0, :, :, :, :].unsqueeze(1))
            pred_object_pose_without_root = pred_object_pose[:, 1:, :, :, :, :]
            pred_object_pose = torch.cat((pred_object_pose_root, pred_object_pose_without_root), 1).view(-1, 63, 5, 13,
                                                                                                         13)
            #pred_object_prob = pred_v_o[:, 3 * self.num_object_control_points:-1, :, :, :]
            pred_object_conf = torch.sigmoid(pred_v_o[:, -1, :, :, :])

            return pred_hand_pose, pred_hand_conf, pred_object_pose, pred_object_conf, pred_hand_vis

        else:
            # x : (batch, 3, 416, 416)
            x = self.features_image(x)

            # extra : (batch, 21, 3)
            extra = extra.view(-1, 21 * 3)
            extra = self.linear_extra(extra)
            extra = self.relu_extra(extra)
            extra = extra.view(batch, -1, 52, 52)
            x = torch.cat([x, extra], dim=1)
            x = self.features_cat(x)

            # x : (batch, 128, 52, 52)
            x = self.features_fuse(x)

            # x : (batch, 512, 13, 13)
            x_1 = self.conv(x)  # input x : (batch, 512, 13, 13)
            x_2 = self.conv_vis(x)

            # x : (batch, 680, 13, 13)
            x = x_1.view(-1, self.hand_vector_size + self.object_vector_size, self.depth_discretization, target_height,
                       target_width)
            pred_hand_vis = x_2.view(-1, 21, self.depth_discretization, target_height, target_width)
            pred_hand_vis = torch.sigmoid(pred_hand_vis)

            pred_v_h = x[:, :self.hand_vector_size, :, :, :]
            pred_v_o = x[:, self.hand_vector_size:, :, :, :]

            # hand specific predictions
            pred_hand_pose = pred_v_h[:, :3 * self.num_hand_control_points, :, :, :]
            pred_hand_pose = pred_hand_pose.view(-1, 21, 3, 5, 13, 13)
            pred_hand_pose_root = torch.sigmoid(pred_hand_pose[:, 0, :, :, :, :].unsqueeze(1))
            pred_hand_pose_without_root = pred_hand_pose[:, 1:, :, :, :, :]
            pred_hand_pose = torch.cat((pred_hand_pose_root, pred_hand_pose_without_root), 1).view(-1, 63, 5, 13, 13)
            #pred_action_prob = pred_v_h[:, 3 * self.num_hand_control_points:-1, :, :, :]
            pred_hand_conf = torch.sigmoid(pred_v_h[:, -1, :, :, :])

            # object specific predictions
            pred_object_pose = pred_v_o[:, :3 * self.num_object_control_points, :, :, :]
            pred_object_pose = pred_object_pose.view(-1, 21, 3, 5, 13, 13)
            pred_object_pose_root = torch.sigmoid(pred_object_pose[:, 0, :, :, :, :].unsqueeze(1))
            pred_object_pose_without_root = pred_object_pose[:, 1:, :, :, :, :]
            pred_object_pose = torch.cat((pred_object_pose_root, pred_object_pose_without_root), 1).view(-1, 63, 5, 13,
                                                                                                         13)
            #pred_object_prob = pred_v_o[:, 3 * self.num_object_control_points:-1, :, :, :]
            pred_object_conf = torch.sigmoid(pred_v_o[:, -1, :, :, :])

            return pred_hand_pose, pred_hand_conf, pred_object_pose, pred_object_conf, pred_hand_vis

    def total_loss(self, pred, true):

        pred_hand_pose, pred_hand_conf, pred_object_pose, pred_object_conf, pred_hand_vis = pred
        true_hand_pose, hand_mask, true_object_pose, object_mask, true_hand_vis = true

        total_pose_loss = self.pose_loss(pred_hand_pose, true_hand_pose, hand_mask) + self.pose_loss(pred_object_pose,
                                                                                                     true_object_pose,
                                                                                                     object_mask)
        total_pose_loss *= 2.

        total_conf_loss = self.conf_loss(pred_hand_conf, pred_hand_pose, true_hand_pose, hand_mask) + self.conf_loss(
            pred_object_conf, pred_object_pose, true_object_pose, object_mask)
        #total_action_loss = self.action_loss(pred_action_prob, true_action_prob, hand_mask)
        #total_object_loss = self.object_loss(pred_object_prob, true_object_prob, object_mask)

        total_vis_loss = self.vis_loss(pred_hand_vis, true_hand_vis)
        #print("loss : ", total_pose_loss, total_conf_loss, total_vis_loss)
        total_loss = total_pose_loss + total_conf_loss + total_vis_loss#+ total_action_loss + total_object_loss

        return total_loss

    def vis_loss(self, pred, true):

        pred = pred.view(-1, 21, 5, 13, 13)
        true = true.view(-1, 21, 5, 13, 13)
        vis_loss = torch.mean(torch.sum(torch.sum(torch.mul(pred - true, pred - true), dim=[1]), dim=[1, 2, 3]))
        return vis_loss

    def pose_loss(self, pred, true, mask):

        pred = pred.view(-1, 21, 3, 5, 13, 13)
        true = true.view(-1, 21, 3, 5, 13, 13)

        masked_pose_loss = torch.mean(torch.sum(mask * torch.sum(torch.mul(pred - true, pred - true), dim=[1, 2]), dim=[1, 2, 3]))
        return masked_pose_loss

    def conf_loss(self, pred_conf, pred, true, mask):

        pred = pred.view(-1, 21, 3, 5, 13, 13)
        true = true.view(-1, 21, 3, 5, 13, 13)

        pred_pixel_x = pred[:, :, 0, :, :, :].unsqueeze(2) * 32 * (1920. / 416)
        pred_pixel_y = pred[:, :, 1, :, :, :].unsqueeze(2) * 32 * (1080. / 416)
        pred_depth = pred[:, :, 2, :, :, :] * 15 * 10

        true_pixel_x = true[:, :, 0, :, :, :].unsqueeze(2) * 32 * (1920. / 416)
        true_pixel_y = true[:, :, 1, :, :, :].unsqueeze(2) * 32 * (1080. / 416)
        true_depth = true[:, :, 2, :, :, :] * 15 * 10

        pixel_x_distance = torch.mul(pred_pixel_x - true_pixel_x, pred_pixel_x - true_pixel_x)
        pixel_y_distance = torch.mul(pred_pixel_y - true_pixel_y, pred_pixel_y - true_pixel_y)
        pixel_distance = torch.sqrt(pixel_x_distance + pixel_y_distance).squeeze(2)
        depth_distance = torch.sqrt(torch.mul(pred_depth - true_depth, pred_depth - true_depth))

        # threshold
        pixel_distance_mask = (pixel_distance < parameters.pixel_threshold).type(torch.cuda.FloatTensor)
        depth_distance_mask = (depth_distance < parameters.depth_threshold).type(torch.cuda.FloatTensor)

        pixel_distance = pixel_distance / (32 * 416 / 1920.)
        depth_disrance = depth_distance / (15 * 10.)

        pixel_distance = torch.from_numpy(pixel_distance.cpu().data.numpy()).cuda()
        depth_distance = torch.from_numpy(depth_distance.cpu().data.numpy()).cuda()

        pixel_conf = torch.exp(parameters.sharpness * (1 - pixel_distance / parameters.pixel_threshold)) / torch.exp(
            parameters.sharpness * (1 - torch.zeros(pixel_distance.size()).cuda()))
        depth_conf = torch.exp(parameters.sharpness * (1 - depth_distance / parameters.depth_threshold)) / torch.exp(
            parameters.sharpness * (1 - torch.zeros(depth_distance.size()).cuda()))

        pixel_conf = torch.mean(pixel_distance_mask * pixel_conf, dim=1)
        depth_conf = torch.mean(depth_distance_mask * depth_conf, dim=1)

        true_conf = 0.5 * (pixel_conf + depth_conf)
        squared_conf_error = torch.mul(pred_conf - true_conf, pred_conf - true_conf)
        exist_conf_error = torch.mean(torch.sum(mask * squared_conf_error, dim=[1, 2, 3]))

        true_conf = torch.zeros(pred_conf.size()).cuda()
        squared_conf_error = torch.mul(pred_conf - true_conf, pred_conf - true_conf)
        no_exist_conf_error = torch.mean(torch.sum((1 - mask) * squared_conf_error, dim=[1, 2, 3]))

        return 5 * exist_conf_error + 0.1 * no_exist_conf_error

    def action_loss(self, pred, true, mask):
        action_ce_loss = self.action_ce_loss(pred, true)
        return torch.mean(torch.sum(mask * action_ce_loss, dim=[1, 2, 3]))

    def object_loss(self, pred, true, mask):
        object_ce_loss = self.object_ce_loss(pred, true)
        return torch.mean(torch.sum(mask * object_ce_loss, dim=[1, 2, 3]))


class UnifiedNetwork(nn.Module):

    def __init__(self):

        super(UnifiedNetwork, self).__init__()

        self.num_hand_control_points = parameters.num_hand_control_points
        self.num_object_control_points = parameters.num_object_control_points
        self.num_actions = parameters.num_actions
        self.num_objects = parameters.num_objects
        self.depth_discretization = parameters.depth_discretization # 5
        
        model = models.resnet18(pretrained=True)

        self.features_image = nn.Sequential(*list(model.children())[:4])
        self.features_fuse = nn.Sequential(*list(model.children())[4:-2])
        #self.features = nn.Sequential(*list(model.children())[:-2])

        self.hand_vector_size = 3 * self.num_hand_control_points + 1 + self.num_actions # 63 + 1 + 4, +1 for confidence
        self.object_vector_size = 3 * self.num_object_control_points + 1 + self.num_objects # 63 + 1 + 4
        self.target_channel_size = self.depth_discretization * ( self.hand_vector_size + self.object_vector_size )  # 5 * 136 = 680

        # prediction layers
        self.conv = nn.Conv2d(512, self.target_channel_size, (3,3), padding=1, bias=True)

        # losses
        self.setup_losses()

        # extra layers
        self.linear = nn.Linear(22*3, 10816)
        self.relu = nn.ReLU(True)



    def setup_losses(self):

        self.action_ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.object_ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, x, extra=None):
        height, width = x.size()[2:]

        assert height == width
        assert height % 32 == 0

        target_height, target_width = int(height / 32), int(width / 32)

        if extra is None:
            # x : (batch, 3, 416, 416)
            x = self.features_image(x)
            # x : (batch, 64, 104, 104)
            x = self.features_fuse(x)
            # x : (batch, 512, 13, 13)
            x = self.conv(x)

            # x : (batch, 680, 13, 13)
            x = x.view(-1, self.hand_vector_size + self.object_vector_size, self.depth_discretization, target_height,
                       target_width)

            # now x : (batch, 136, 5, 13, 13)
            pred_v_h = x[:, :self.hand_vector_size, :, :, :]
            pred_v_o = x[:, self.hand_vector_size:, :, :, :]

            # hand specific predictions
            pred_hand_pose = pred_v_h[:, :3 * self.num_hand_control_points, :, :, :]
            pred_hand_pose = pred_hand_pose.view(-1, 21, 3, 5, 13, 13)
            pred_hand_pose_root = torch.sigmoid(pred_hand_pose[:, 0, :, :, :, :].unsqueeze(1))
            pred_hand_pose_without_root = pred_hand_pose[:, 1:, :, :, :, :]
            pred_hand_pose = torch.cat((pred_hand_pose_root, pred_hand_pose_without_root), 1).view(-1, 63, 5, 13, 13)
            pred_action_prob = pred_v_h[:, 3 * self.num_hand_control_points:-1, :, :, :]
            pred_hand_conf = torch.sigmoid(pred_v_h[:, -1, :, :, :])

            # object specific predictions
            pred_object_pose = pred_v_o[:, :3 * self.num_object_control_points, :, :, :]
            pred_object_pose = pred_object_pose.view(-1, 21, 3, 5, 13, 13)
            pred_object_pose_root = torch.sigmoid(pred_object_pose[:, 0, :, :, :, :].unsqueeze(1))
            pred_object_pose_without_root = pred_object_pose[:, 1:, :, :, :, :]
            pred_object_pose = torch.cat((pred_object_pose_root, pred_object_pose_without_root), 1).view(-1, 63, 5, 13,
                                                                                                         13)
            pred_object_prob = pred_v_o[:, 3 * self.num_object_control_points:-1, :, :, :]
            pred_object_conf = torch.sigmoid(pred_v_o[:, -1, :, :, :])

            return pred_hand_pose, pred_action_prob, pred_hand_conf, pred_object_pose, pred_object_prob, pred_object_conf

        else:
            # x : (batch, 3, 416, 416)
            x = self.features_image(x)

            # extra : (batch, 22, 3)
            extra = extra.view(-1, 22 * 3)
            extra = self.linear(extra)
            extra = self.relu(extra)
            extra = extra.view(-1, 104, 104)
            ###
            # view 하는 과정에서 batch 간의 data가 안 섞이나?
            ###

            x[:, 0, :, :] = extra

            # x : (batch, 64, 104, 104)
            x = self.features_fuse(x)

            # x : (batch, 512, 13, 13)
            x = self.conv(x)

            # x : (batch, 680, 13, 13)
            x = x.view(-1, self.hand_vector_size + self.object_vector_size, self.depth_discretization, target_height,
                       target_width)

            # now x : (batch, 136, 5, 13, 13)

            pred_v_h = x[:, :self.hand_vector_size, :, :, :]
            pred_v_o = x[:, self.hand_vector_size:, :, :, :]

            # hand specific predictions
            pred_hand_pose = pred_v_h[:, :3 * self.num_hand_control_points, :, :, :]
            pred_hand_pose = pred_hand_pose.view(-1, 21, 3, 5, 13, 13)
            pred_hand_pose_root = torch.sigmoid(pred_hand_pose[:, 0, :, :, :, :].unsqueeze(1))
            pred_hand_pose_without_root = pred_hand_pose[:, 1:, :, :, :, :]
            pred_hand_pose = torch.cat((pred_hand_pose_root, pred_hand_pose_without_root), 1).view(-1, 63, 5, 13, 13)
            pred_action_prob = pred_v_h[:, 3 * self.num_hand_control_points:-1, :, :, :]
            pred_hand_conf = torch.sigmoid(pred_v_h[:, -1, :, :, :])

            # object specific predictions
            pred_object_pose = pred_v_o[:, :3 * self.num_object_control_points, :, :, :]
            pred_object_pose = pred_object_pose.view(-1, 21, 3, 5, 13, 13)
            pred_object_pose_root = torch.sigmoid(pred_object_pose[:, 0, :, :, :, :].unsqueeze(1))
            pred_object_pose_without_root = pred_object_pose[:, 1:, :, :, :, :]
            pred_object_pose = torch.cat((pred_object_pose_root, pred_object_pose_without_root), 1).view(-1, 63, 5, 13,
                                                                                                         13)
            pred_object_prob = pred_v_o[:, 3 * self.num_object_control_points:-1, :, :, :]
            pred_object_conf = torch.sigmoid(pred_v_o[:, -1, :, :, :])

            return pred_hand_pose, pred_action_prob, pred_hand_conf, pred_object_pose, pred_object_prob, pred_object_conf


    def total_loss(self, pred, true):

        pred_hand_pose, pred_action_prob, pred_hand_conf, pred_object_pose, pred_object_prob, pred_object_conf = pred
        true_hand_pose, true_action_prob, hand_mask, true_object_pose, true_object_prob, object_mask = true
        
        total_pose_loss = self.pose_loss(pred_hand_pose, true_hand_pose, hand_mask) + self.pose_loss(pred_object_pose, true_object_pose, object_mask)
        total_conf_loss = self.conf_loss(pred_hand_conf, pred_hand_pose, true_hand_pose, hand_mask) + self.conf_loss(pred_object_conf, pred_object_pose, true_object_pose, object_mask)
        total_action_loss = self.action_loss(pred_action_prob, true_action_prob, hand_mask)
        total_object_loss = self.object_loss(pred_object_prob, true_object_prob, object_mask)

        total_loss = total_pose_loss + total_action_loss + total_object_loss + total_conf_loss

        return total_loss

    def pose_loss(self, pred, true, mask):

        pred = pred.view(-1, 21, 3, 5, 13, 13)
        true = true.view(-1, 21, 3, 5, 13, 13)
        masked_pose_loss = torch.mean(torch.sum(mask * torch.sum(torch.mul(pred - true, pred - true), dim=[1,2]), dim=[1,2,3]))
        return masked_pose_loss
    
    def conf_loss(self, pred_conf, pred, true, mask):
        
        pred = pred.view(-1, 21, 3, 5, 13, 13)
        true = true.view(-1, 21, 3, 5, 13, 13)

        pred_pixel_x = pred[:, :, 0, :, :, :].unsqueeze(2) * 32 * (1920. / 416)
        pred_pixel_y = pred[:, :, 1, :, :, :].unsqueeze(2) * 32 * (1080. / 416)
        pred_depth = pred[:, :, 2, :, :, :] * 15 * 10

        true_pixel_x = true[:, :, 0, :, :, :].unsqueeze(2) * 32 * (1920. / 416)
        true_pixel_y = true[:, :, 1, :, :, :].unsqueeze(2) * 32 * (1080. / 416)
        true_depth = true[:, :, 2, :, :, :] * 15 * 10

        pixel_x_distance = torch.mul(pred_pixel_x - true_pixel_x, pred_pixel_x - true_pixel_x)
        pixel_y_distance = torch.mul(pred_pixel_y - true_pixel_y, pred_pixel_y - true_pixel_y)
        pixel_distance = torch.sqrt(pixel_x_distance + pixel_y_distance).squeeze(2)
        depth_distance = torch.sqrt(torch.mul(pred_depth - true_depth, pred_depth - true_depth))
        
        # threshold
        pixel_distance_mask = (pixel_distance < parameters.pixel_threshold).type(torch.cuda.FloatTensor)
        depth_distance_mask = (depth_distance < parameters.depth_threshold).type(torch.cuda.FloatTensor)

        pixel_distance = pixel_distance / (32 * 416 / 1920.)
        depth_disrance = depth_distance / (15 * 10.)

        pixel_distance = torch.from_numpy(pixel_distance.cpu().data.numpy()).cuda()
        depth_distance = torch.from_numpy(depth_distance.cpu().data.numpy()).cuda()

        pixel_conf = torch.exp(parameters.sharpness * (1 - pixel_distance / parameters.pixel_threshold)) / torch.exp(parameters.sharpness * (1 - torch.zeros(pixel_distance.size()).cuda()))
        depth_conf = torch.exp(parameters.sharpness * (1 - depth_distance / parameters.depth_threshold)) / torch.exp(parameters.sharpness * (1 - torch.zeros(depth_distance.size()).cuda()))

        pixel_conf = torch.mean(pixel_distance_mask * pixel_conf, dim=1)
        depth_conf = torch.mean(depth_distance_mask * depth_conf, dim=1)

        true_conf = 0.5 * (pixel_conf + depth_conf)
        squared_conf_error = torch.mul(pred_conf - true_conf, pred_conf - true_conf)
        exist_conf_error = torch.mean(torch.sum(mask * squared_conf_error, dim=[1,2,3]))

        true_conf = torch.zeros(pred_conf.size()).cuda()
        squared_conf_error = torch.mul(pred_conf - true_conf, pred_conf - true_conf)
        no_exist_conf_error = torch.mean(torch.sum((1 - mask) * squared_conf_error, dim=[1,2,3]))

        return 5 * exist_conf_error + 0.1 * no_exist_conf_error
        
    def action_loss(self, pred, true, mask):
        action_ce_loss = self.action_ce_loss(pred, true)
        return torch.mean(torch.sum(mask * action_ce_loss, dim=[1,2,3]))

    def object_loss(self, pred, true, mask):
        object_ce_loss = self.object_ce_loss(pred, true)
        return torch.mean(torch.sum(mask * object_ce_loss, dim=[1,2,3]))

if __name__ == '__main__':

    model = UnifiedNetwork()
    x = torch.randn(32, 3, 416, 416)

    true = torch.randn(32, 76, 5, 13, 13), torch.randn(32, 74, 5, 13, 13)

    pred = model(x)
    
    true_hand_pose = torch.randn(32, 3 * parameters.num_hand_control_points, 5, 13, 13)
    true_action_prob = torch.empty(32, 5, 13, 13, dtype=torch.long).random_(parameters.num_actions)
    hand_mask = torch.zeros(5, 13, 13, dtype=torch.float32)
    hand_mask[0, 0, 0] = 1.

    true_object_pose = torch.randn(32, 3 * parameters.num_object_control_points, 5, 13, 13)
    true_object_prob = torch.empty(32, 5, 13, 13, dtype=torch.long).random_(parameters.num_objects)
    object_mask = torch.zeros(5, 13, 13, dtype=torch.float32)
    object_mask[0, 0, 0] = 1.

    true = true_hand_pose, true_action_prob, hand_mask, true_object_pose, true_object_prob, object_mask

    print(model.total_loss(pred, true))
    
