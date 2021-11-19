import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from cfg import parameters

NUM_hand_control_points = parameters.num_hand_control_points
NUM_object_control_points = parameters.num_object_control_points
NUM_objects = parameters.num_objects
DEPTH_discretization = parameters.depth_discretization  # 5
NUM_visibility = 21


class Network_utils():
    def __init__(self):
        self.IMAGE_WIDTH = None
        self.IMAGE_HEIGHT = None

    def update_parameter(self, img_width, img_height):
        self.IMAGE_WIDTH = img_width
        self.IMAGE_HEIGHT = img_height

    def setup_losses(self):
        self.action_ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.object_ce_loss = nn.CrossEntropyLoss(reduction='none')

    def total_loss(self, pred, true, flag_log=False):

        pred_hand_pose, pred_hand_conf, pred_object_pose, pred_object_conf, pred_hand_vis = pred
        true_hand_pose, hand_mask, true_object_pose, object_mask, true_hand_vis = true

        total_pose_loss = self.pose_loss(pred_hand_pose, true_hand_pose, hand_mask) + self.pose_loss(pred_object_pose,
                                                                                                     true_object_pose,
                                                                                                     object_mask)
        # total_pose_loss *= 2.

        total_conf_loss = self.conf_loss(pred_hand_conf, pred_hand_pose, true_hand_pose, hand_mask) + self.conf_loss(
            pred_object_conf, pred_object_pose, true_object_pose, object_mask)
        #total_action_loss = self.action_loss(pred_action_prob, true_action_prob, hand_mask)
        #total_object_loss = self.object_loss(pred_object_prob, true_object_prob, object_mask)

        total_vis_loss = self.vis_loss(pred_hand_vis, true_hand_vis)
        if flag_log:
            print("specific losses (pose/conf/vis) : ", total_pose_loss.data, total_conf_loss.data, total_vis_loss.data)
        total_loss = total_pose_loss + total_conf_loss + total_vis_loss#+ total_action_loss + total_object_loss

        return total_loss

    def total_loss_FHAD(self, pred, true):

        pred_hand_pose, pred_hand_conf, pred_object_pose, pred_object_conf, _ = pred
        true_hand_pose, hand_mask, true_object_pose, object_mask = true

        total_pose_loss = self.pose_loss(pred_hand_pose, true_hand_pose, hand_mask) + self.pose_loss(pred_object_pose,
                                                                                                     true_object_pose,
                                                                                                     object_mask)
        # total_pose_loss *= 2.

        total_conf_loss = self.conf_loss(pred_hand_conf, pred_hand_pose, true_hand_pose, hand_mask) + self.conf_loss(
            pred_object_conf, pred_object_pose, true_object_pose, object_mask)
        #total_action_loss = self.action_loss(pred_action_prob, true_action_prob, hand_mask)
        #total_object_loss = self.object_loss(pred_object_prob, true_object_prob, object_mask)

        total_loss = total_pose_loss + total_conf_loss#+ total_action_loss + total_object_loss

        return total_loss

    def total_loss_FreiHAND(self, pred, true):

        pred_hand_pose, pred_hand_conf, _, _, _ = pred
        true_hand_pose, hand_mask, _ = true

        total_pose_loss = self.pose_loss(pred_hand_pose, true_hand_pose, hand_mask)

        # total_pose_loss *= 2.

        total_conf_loss = self.conf_loss(pred_hand_conf, pred_hand_pose, true_hand_pose, hand_mask)

        #total_action_loss = self.action_loss(pred_action_prob, true_action_prob, hand_mask)
        #total_object_loss = self.object_loss(pred_object_prob, true_object_prob, object_mask)

        total_loss = total_pose_loss + total_conf_loss#+ total_action_loss + total_object_loss

        return total_loss

    def total_loss_Obman(self, pred, true):

        pred_hand_pose, pred_hand_conf, _, _, _ = pred
        true_hand_pose, hand_mask = true

        total_pose_loss = self.pose_loss(pred_hand_pose, true_hand_pose, hand_mask)

        # total_pose_loss *= 2.

        total_conf_loss = self.conf_loss(pred_hand_conf, pred_hand_pose, true_hand_pose, hand_mask)

        #total_action_loss = self.action_loss(pred_action_prob, true_action_prob, hand_mask)
        #total_object_loss = self.object_loss(pred_object_prob, true_object_prob, object_mask)

        total_loss = total_pose_loss + total_conf_loss#+ total_action_loss + total_object_loss

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

        pred_pixel_x = pred[:, :, 0, :, :, :].unsqueeze(2) * 32 * (self.IMAGE_WIDTH / 416.)
        pred_pixel_y = pred[:, :, 1, :, :, :].unsqueeze(2) * 32 * (self.IMAGE_HEIGHT / 416.)
        pred_depth = pred[:, :, 2, :, :, :] * 25 * 10

        true_pixel_x = true[:, :, 0, :, :, :].unsqueeze(2) * 32 * (self.IMAGE_WIDTH / 416.)
        true_pixel_y = true[:, :, 1, :, :, :].unsqueeze(2) * 32 * (self.IMAGE_HEIGHT / 416.)
        true_depth = true[:, :, 2, :, :, :] * 25 * 10

        pixel_x_distance = torch.mul(pred_pixel_x - true_pixel_x, pred_pixel_x - true_pixel_x)
        pixel_y_distance = torch.mul(pred_pixel_y - true_pixel_y, pred_pixel_y - true_pixel_y)
        pixel_distance = torch.sqrt(pixel_x_distance + pixel_y_distance).squeeze(2)
        depth_distance = torch.sqrt(torch.mul(pred_depth - true_depth, pred_depth - true_depth))

        # threshold
        pixel_distance_mask = (pixel_distance < parameters.pixel_threshold).type(torch.cuda.FloatTensor)
        depth_distance_mask = (depth_distance < parameters.depth_threshold).type(torch.cuda.FloatTensor)

        pixel_distance = pixel_distance / (32 * 416. / self.IMAGE_WIDTH)
        depth_disrance = depth_distance / (25 * 10.)

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

    def check_status(self):
        assert self.IMAGE_WIDTH is not None, 'define image size of target dataset'
        print("current image size setting : [{}, {}]".format(self.IMAGE_WIDTH, self.IMAGE_HEIGHT))

    def split_result(self, pred_v_h, pred_v_o):
        # hand specific predictions
        pred_hand_pose = pred_v_h[:, :3 * NUM_hand_control_points, :, :, :]
        pred_hand_pose = pred_hand_pose.view(-1, 21, 3, 5, 13, 13)
        pred_hand_pose_root = torch.sigmoid(pred_hand_pose[:, 0, :, :, :, :].unsqueeze(1))
        pred_hand_pose_without_root = pred_hand_pose[:, 1:, :, :, :, :]
        pred_hand_pose = torch.cat((pred_hand_pose_root, pred_hand_pose_without_root), 1).view(-1, 63, 5, 13, 13)
        # pred_action_prob = pred_v_h[:, 3 * NUM_hand_control_points:-1, :, :, :]
        pred_hand_conf = torch.sigmoid(pred_v_h[:, -1, :, :, :])

        pred_hand_vis = pred_v_h[:, 3 * NUM_hand_control_points:-1, :, :, :]  # (-1, 21, 5, 13, 13)
        pred_hand_vis = torch.sigmoid(pred_hand_vis)

        # object specific predictions
        pred_object_pose = pred_v_o[:, :3 * NUM_object_control_points, :, :, :]
        pred_object_pose = pred_object_pose.view(-1, 21, 3, 5, 13, 13)
        pred_object_pose_root = torch.sigmoid(pred_object_pose[:, 0, :, :, :, :].unsqueeze(1))
        pred_object_pose_without_root = pred_object_pose[:, 1:, :, :, :, :]
        pred_object_pose = torch.cat((pred_object_pose_root, pred_object_pose_without_root), 1).view(-1, 63, 5, 13,
                                                                                                     13)
        # pred_object_prob = pred_v_o[:, 3 * self.num_object_control_points:-1, :, :, :]
        pred_object_conf = torch.sigmoid(pred_v_o[:, -1, :, :, :])

        return pred_hand_pose, pred_hand_conf, pred_object_pose, pred_object_conf, pred_hand_vis


class UnifiedNet_res34_lowconcat(nn.Module, Network_utils):

    def __init__(self):
        super(UnifiedNet_res34_lowconcat, self).__init__()

        model = models.resnet34(pretrained=True)

        self.features_image = nn.Sequential(*list(model.children())[:8])
        self.features_fuse = nn.Sequential(*list(model.children())[8:-2])
        # self.features = nn.Sequential(*list(model.children())[:-2])
        self.features_cat = nn.Conv2d(544, 512, 3, padding=1)

        self.hand_vector_size = 3 * NUM_hand_control_points + 1 + NUM_visibility  # 63 + 1 + 21, (1 for confidence)
        self.object_vector_size = 3 * NUM_object_control_points + 1 + NUM_objects  # 63 + 1 + 4
        self.target_channel_size = DEPTH_discretization * (
                    self.hand_vector_size + self.object_vector_size)  # 5 * 153 = 765

        # prediction layers
        self.conv = nn.Conv2d(512, self.target_channel_size, (3, 3), padding=1, bias=True)

        # losses
        self.setup_losses()

        # extra layers
        self.linear_extra = nn.Linear(84, 5408)
        self.relu_extra = nn.ReLU(True)

        # initialize specific layers
        torch.nn.init.xavier_uniform_(self.features_cat.weight)
        torch.nn.init.xavier_uniform_(self.linear_extra.weight)

    def forward(self, x, extra=None):
        batch = x.size()[0]
        height, width = x.size()[2:]

        assert height == width
        assert height % 32 == 0

        target_height, target_width = int(height / 32), int(width / 32)

        if extra is None:
            extra = torch.zeros(batch, 4, 21).cuda()

        # x : (batch, 3, 416, 416)
        x = self.features_image(x)

        # extra : (batch, 63, 5, 13, 13)
        extra = extra.view(-1, 4 * 21)
        extra = self.linear_extra(extra)
        extra = self.relu_extra(extra)
        extra = extra.view(batch, -1, 13, 13)
        x = torch.cat([x, extra], dim=1)
        x = self.features_cat(x)

        # x : (batch, 128, 52, 52)
        x = self.features_fuse(x)

        # x : (batch, 512, 13, 13)
        x = self.conv(x)  # input x : (batch, 512, 13, 13)

        # x : (batch, 765, 13, 13)
        x = x.view(-1, self.hand_vector_size + self.object_vector_size, DEPTH_discretization, target_height,
                   target_width)

        # now x : (batch, 153, 5, 13, 13)
        pred_v_h = x[:, :self.hand_vector_size, :, :, :]
        pred_v_o = x[:, self.hand_vector_size:, :, :, :]

        return self.split_result(pred_v_h, pred_v_o)
        # return pred_hand_pose, pred_hand_conf, pred_object_pose, pred_object_conf, pred_hand_vis


class UnifiedNet_res34(nn.Module, Network_utils):

    def __init__(self):

        super(UnifiedNet_res34, self).__init__()

        model = models.resnet34(pretrained=True)

        self.features_image = nn.Sequential(*list(model.children())[:4])
        self.features_fuse = nn.Sequential(*list(model.children())[4:-2])
        # self.features = nn.Sequential(*list(model.children())[:-2])
        self.features_cat = nn.Conv2d(65, 64, 3, padding=1)

        self.hand_vector_size = 3 * NUM_hand_control_points + 1 + NUM_visibility  # 63 + 1 + 21, (1 for confidence)
        self.object_vector_size = 3 * NUM_object_control_points + 1 + NUM_objects  # 63 + 1 + 4
        self.target_channel_size = DEPTH_discretization * (
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

    def forward(self, x, extra=None):
        batch = x.size()[0]
        height, width = x.size()[2:]

        assert height == width
        assert height % 32 == 0

        target_height, target_width = int(height / 32), int(width / 32)

        if extra is None:
            extra = torch.zeros(batch, 4, 21).cuda()

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
        x = x.view(-1, self.hand_vector_size + self.object_vector_size, DEPTH_discretization, target_height,
                   target_width)

        # x : (batch, 153, 5, 13, 13)
        pred_v_h = x[:, :self.hand_vector_size, :, :, :]
        pred_v_o = x[:, self.hand_vector_size:, :, :, :]

        return self.split_result(pred_v_h, pred_v_o)


class UnifiedNet_res18_lowconcat(nn.Module, Network_utils):

    def __init__(self):

        super(UnifiedNet_res18_lowconcat, self).__init__()

        model = models.resnet18(pretrained=True)

        self.features_image = nn.Sequential(*list(model.children())[:8])
        self.features_fuse = nn.Sequential(*list(model.children())[8:-2])
        # self.features = nn.Sequential(*list(model.children())[:-2])
        self.features_cat = nn.Conv2d(544, 512, 3, padding=1)

        self.hand_vector_size = 3 * NUM_hand_control_points + 1 + NUM_visibility  # 63 + 1 + 21, (1 for confidence)
        self.object_vector_size = 3 * NUM_object_control_points + 1 + NUM_objects  # 63 + 1 + 4
        self.target_channel_size = DEPTH_discretization * (
                    self.hand_vector_size + self.object_vector_size)  # 5 * 153 = 765

        # prediction layers
        self.conv = nn.Conv2d(512, self.target_channel_size, (3, 3), padding=1, bias=True)

        # losses
        self.setup_losses()

        # extra layers
        self.linear_extra = nn.Linear(84, 5408)
        self.relu_extra = nn.ReLU(True)

        # initialize specific layers
        torch.nn.init.xavier_uniform_(self.features_cat.weight)
        torch.nn.init.xavier_uniform_(self.linear_extra.weight)

    def forward(self, x, extra=None):
        batch = x.size()[0]
        height, width = x.size()[2:]

        assert height == width
        assert height % 32 == 0

        target_height, target_width = int(height / 32), int(width / 32)

        if extra is None:
            extra = torch.zeros(batch, 4, 21).cuda()

        # x : (batch, 3, 416, 416)
        x = self.features_image(x)

        # extra : (batch, 63, 5, 13, 13)
        extra = extra.view(-1, 4 * 21)
        extra = self.linear_extra(extra)
        extra = self.relu_extra(extra)
        extra = extra.view(batch, -1, 13, 13)
        x = torch.cat([x, extra], dim=1)
        x = self.features_cat(x)

        # x : (batch, 128, 52, 52)
        x = self.features_fuse(x)

        # x : (batch, 512, 13, 13)
        x = self.conv(x)  # input x : (batch, 512, 13, 13)

        # x : (batch, 765, 13, 13)
        x = x.view(-1, self.hand_vector_size + self.object_vector_size, DEPTH_discretization, target_height,
                   target_width)

        # x : (batch, 153, 5, 13, 13)
        pred_v_h = x[:, :self.hand_vector_size, :, :, :]
        pred_v_o = x[:, self.hand_vector_size:, :, :, :]

        return self.split_result(pred_v_h, pred_v_o)


class UnifiedNet_res18(nn.Module, Network_utils):

    def __init__(self):
        super(UnifiedNet_res18, self).__init__()

        model = models.resnet18(pretrained=True)

        self.features_image = nn.Sequential(*list(model.children())[:4])
        self.features_fuse = nn.Sequential(*list(model.children())[4:-2])
        # self.features = nn.Sequential(*list(model.children())[:-2])
        self.features_cat = nn.Conv2d(65, 64, 3, padding=1)

        self.hand_vector_size = 3 * NUM_hand_control_points + 1 + NUM_visibility  # 63 + 1 + 21, (1 for confidence)
        self.object_vector_size = 3 * NUM_object_control_points + 1 + NUM_objects  # 63 + 1 + 4
        self.target_channel_size = DEPTH_discretization * (
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


    def forward(self, x, extra=None):
        batch = x.size()[0]
        height, width = x.size()[2:]

        assert height == width
        assert height % 32 == 0

        target_height, target_width = int(height / 32), int(width / 32)

        if extra is None:
            extra = torch.zeros(batch, 4, 21).cuda()

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
        x = x.view(-1, self.hand_vector_size + self.object_vector_size, DEPTH_discretization, target_height,
                   target_width)

        # x : (batch, 153, 5, 13, 13)
        pred_v_h = x[:, :self.hand_vector_size, :, :, :]
        pred_v_o = x[:, self.hand_vector_size:, :, :, :]

        return self.split_result(pred_v_h, pred_v_o)


class UnifiedNet_res18_noextra(nn.Module, Network_utils):

    def __init__(self):
        super(UnifiedNet_res18_noextra, self).__init__()

        model = models.resnet18(pretrained=True)

        self.features = nn.Sequential(*list(model.children())[:-2])

        self.hand_vector_size = 3 * NUM_hand_control_points + 1 + NUM_visibility  # 63 + 1 + 21, (1 for confidence)
        self.object_vector_size = 3 * NUM_object_control_points + 1 + NUM_objects  # 63 + 1 + 4
        self.target_channel_size = DEPTH_discretization * (
                    self.hand_vector_size + self.object_vector_size)  # 5 * 153 = 765

        # prediction layers
        self.conv = nn.Conv2d(512, self.target_channel_size, (3, 3), padding=1, bias=True)

        # losses
        self.setup_losses()


    def forward(self, x):
        batch = x.size()[0]
        height, width = x.size()[2:]

        assert height == width
        assert height % 32 == 0

        target_height, target_width = int(height / 32), int(width / 32)

        x = self.features(x)
        x = self.conv(x)
        # x : (batch, 765, 13, 13)
        x = x.view(-1, self.hand_vector_size + self.object_vector_size, DEPTH_discretization, target_height,
                   target_width)

        # x : (batch, 153, 5, 13, 13)
        pred_v_h = x[:, :self.hand_vector_size, :, :, :]
        pred_v_o = x[:, self.hand_vector_size:, :, :, :]

        return self.split_result(pred_v_h, pred_v_o)


if __name__ == '__main__':
    print("main loop")
    """
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
    """
