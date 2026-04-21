import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import numpy as np
from einops import rearrange
import copy
from sklearn.metrics import mutual_info_score
from torch import Tensor
import random
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import (Adj, Size, OptTensor, PairTensor)
from typing import Union, Tuple, Optional

from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

# import preprocess.SEED_pretrain as SEED_pre
from utils import *
import torchvision.models as models
import logging


# 执行数据的预处理，包括att将数据升维和resnet提取数据特征
# ConvNet encoder
class EncoderNet(nn.Module):
    def __init__(self, args):
        super(EncoderNet, self).__init__()
        self.args = args
        self.resnet_embed = 256
        self.backbone_output =  self.resnet_embed * 2
        self.num_classes = getattr(args, "num_classes", 2)
        self.dataset_config = get_g2g_dataset_config(getattr(args, "dataset", ""))
        self.eeg_nodes = self.dataset_config["eeg_nodes"]
        self.eeg_feature_dim = self.dataset_config["eeg_feature_dim"]
        self.eye_nodes = self.dataset_config["eye_nodes"]
        self.eye_feature_dim = self.dataset_config["eye_feature_dim"]
        self.eeg_flat_dim = self.eeg_nodes * self.eeg_feature_dim
        self.eye_flat_dim = self.eye_nodes * self.eye_feature_dim

        self.relationAwareness = RelationAwareness(args = self.args)
        self.rand_order = random_1D_node(2, self.eeg_nodes)

        # define selected backbone
        self.backbone = ResNet50()

        # get node location
        self.location = torch.from_numpy(self.dataset_config["coordinates"]).to(self.args.device)

        self.l_relu = nn.LeakyReLU(0.1)
        self.bn = nn.BatchNorm1d(self.backbone_output)
        self.bn_2D = nn.BatchNorm2d(12)
        self.dropout = nn.Dropout(0.3)
        self.tanh = nn.Tanh()

        self.mlp_0 = nn.Linear(512, self.backbone_output)
        self.mlp_1 = nn.Linear(self.backbone_output, self.num_classes)


    def forward(self, x):

        # output_original =copy.deepcopy(x.view(x.size(0), -1))
        # 随机排列运行分支
        ######################################################
        ran_list = []
        expected_dim = self.eeg_flat_dim + self.eye_flat_dim
        if x.shape[1] != expected_dim:
            raise ValueError(
                f"G2G expected flattened feature dim {expected_dim}, got {x.shape[1]} "
                f"for dataset {getattr(self.args, 'dataset', 'unknown')}"
            )
        for index in range(2):
            x_eeg = x[:, :self.eeg_flat_dim]
            x_eye = x[:, self.eeg_flat_dim:self.eeg_flat_dim + self.eye_flat_dim]

            x_eeg = rearrange(x_eeg, 'b (h c) -> b h c', h=self.eeg_nodes)
            x_eye = rearrange(x_eye, 'b (h c) -> b h c', h=self.eye_nodes)

            x_random = x_eeg[:, self.rand_order[index], :]
            coor_random = self.location[self.rand_order[index], :]
            x_ = self.relationAwareness(x_random, coor_random, x_eye) # (batch_size, 62, 62, 3)/ (32,32,3)

            ran_list.append(x_)

        x_ = torch.cat(tuple(ran_list), 1)  # (batch_size, self.args.rand_ali_num*self.head, N, N)
        x_ = self.bn_2D(x_)

        output = self.backbone(x_)

        x = self.dropout(output)
        x = self.mlp_0(x)
        x = self.l_relu(x)
        x = self.bn(x)
        x = self.mlp_1(x)

        return x

# 运行位置相关的自注意力
class RelationAwareness(nn.Module):
    def __init__(self, args):
        super(RelationAwareness, self).__init__()

        self.head = 6
        self.input_size = 5 # eeg input size on each electrode, 5
        self.location_size = 3 # 3
        self.expand_size = 10 # expand eeg, eye, and location to the same dimen, 10

        self.location_em = nn.Linear(self.location_size, self.head*self.expand_size) # 3 --> 6*10
        self.data_em = nn.Linear(self.input_size, self.head*self.expand_size) # 5 --> 6*10
        self.eye_em = nn.Linear(10, self.head*self.expand_size) # 10 --> 6*10
        self.relu = nn.ReLU()
        self.args = args

        self.a = nn.Parameter(torch.empty(size=(2*self.expand_size, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, feature, location, eye):

        feature_embed = self.data_em(feature)
        location_embed = self.location_em(location)
        feature_local_embed = self.relu(feature_embed + location_embed)

        eye_embed = self.relu(self.eye_em(eye))
        eeg_eye_embed = torch.cat([feature_local_embed, eye_embed], 1)

        feature_ = rearrange(eeg_eye_embed, "b n (h d) -> b h n d", h=self.head)

        two_d_feature = self.cal_att_matrix(feature_)
        return two_d_feature

    def cal_att_matrix(self, feature):

        data = []
        batch_size, head,  N = feature.size(0), feature.size(1), feature.size(2)
        Wh1 = torch.matmul(feature, self.a[:self.expand_size, :])
        Wh2 = torch.matmul(feature, self.a[self.expand_size:, :])
        # broadcast add
        Wh2_T = rearrange(Wh2, "b n h d -> b n d h")
        e = Wh1 + Wh2_T
        return e


class ConvNet(nn.Module):
    def __init__(self, emb_size, args, cifar_flag=False):
        super(ConvNet, self).__init__()
        # set size
        self.hidden = 128
        self.last_hidden = self.hidden * 16 if not cifar_flag else self.hidden
        # self.last_hidden = self.hidden * 1 if not cifar_flag else self.hidden
        self.emb_size = emb_size
        self.args = args

        # set layers
        self.conv_1 = nn.Sequential(nn.Conv2d(in_channels=12,
                                              out_channels=self.hidden,
                                              kernel_size=3,
                                              padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=self.hidden),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv_2 = nn.Sequential(nn.Conv2d(in_channels=self.hidden,
                                              out_channels=int(self.hidden*1.5),
                                              kernel_size=3,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=int(self.hidden*1.5)),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv_3 = nn.Sequential(nn.Conv2d(in_channels=int(self.hidden*1.5),
                                              out_channels=self.hidden*2,
                                              kernel_size=3,
                                              padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=self.hidden * 2),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    nn.Dropout2d(0.4))
        self.max = nn.MaxPool2d(kernel_size=2)

        self.conv_4 = nn.Sequential(nn.Conv2d(in_channels=self.hidden*2,
                                              out_channels=self.hidden*4,
                                              kernel_size=3,
                                              padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=self.hidden * 4),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    nn.Dropout2d(0.5))
        self.layer_second = nn.Sequential(nn.Linear(in_features=self.last_hidden * 2 ,
                                                    out_features=self.emb_size, bias=True),
                                          nn.BatchNorm1d(self.emb_size))
        self.layer_last = nn.Sequential(nn.Linear(in_features=self.last_hidden * 4 ,
                                                  out_features=self.emb_size, bias=True),
                                        nn.BatchNorm1d(self.emb_size))

    def forward(self, input_data):
        out_1 = self.conv_1(input_data)
        out_2 = self.conv_2(out_1)
        out_3 = self.conv_3(out_2)
        output_data = self.conv_4(out_3)
        output_data0 = self.max(out_3)
        out1 = self.layer_last(output_data.view(output_data.size(0), -1))
        out2 = self.layer_second(output_data0.view(output_data0.size(0), -1))

        out = torch.cat((out1, out2), dim=1)  # (batch_size, 256)

        return out


class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        self.resnet50.conv1 = nn.Conv2d(12, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet50.fc = nn.Linear(2048, 512)

    def forward(self, x):
        x = self.resnet50(x)
        return x


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        self.resnet18.conv1 = nn.Conv2d(12, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet18.fc = nn.Linear(512, 512)

    def forward(self, x):
        x = self.resnet18(x)
        return x


def random_1D_node(num, node_num):

    rand_lists = []
    for index in range(num):
        grand_list = [i for i in range(node_num)]
        random.shuffle(grand_list)
        rand_tensor = torch.tensor(grand_list).view(1, node_num)
        rand_lists.append(rand_tensor)

    rand_torch = torch.cat(tuple(rand_lists), 0)
    return rand_torch


def get_g2g_dataset_config(dataset):
    dataset_name = dataset.lower()
    if dataset_name.startswith("deap"):
        return {
            "eeg_nodes": 32,
            "eeg_feature_dim": 5,
            "eye_nodes": 8,
            "eye_feature_dim": 10,
            "coordinates": return_coordinates_deap(),
        }
    if dataset_name.startswith("seed"):
        return {
            "eeg_nodes": 62,
            "eeg_feature_dim": 5,
            "eye_nodes": 6,
            "eye_feature_dim": 10,
            "coordinates": return_coordinates(),
        }
    raise ValueError(f"Unsupported dataset for G2G: {dataset}")


def return_coordinates():
    """
    Node location for SEED, SEED4, SEED5, MPED
    """
    m1 = [(-2.285379, 10.372299, 4.564709),
          (0.687462, 10.931931, 4.452579),
          (3.874373, 9.896583, 4.368097),
          (-2.82271, 9.895013, 6.833403),
          (4.143959, 9.607678, 7.067061),

          (-6.417786, 6.362997, 4.476012),
          (-5.745505, 7.282387, 6.764246),
          (-4.248579, 7.990933, 8.73188),
          (-2.046628, 8.049909, 10.162745),
          (0.716282, 7.836015, 10.88362),
          (3.193455, 7.889754, 10.312743),
          (5.337832, 7.691511, 8.678795),
          (6.842302, 6.643506, 6.300108),
          (7.197982, 5.671902, 4.245699),

          (-7.326021, 3.749974, 4.734323),
          (-6.882368, 4.211114, 7.939393),
          (-4.837038, 4.672796, 10.955297),
          (-2.677567, 4.478631, 12.365311),
          (0.455027, 4.186858, 13.104445),
          (3.654295, 4.254963, 12.205945),
          (5.863695, 4.275586, 10.714709),
          (7.610693, 3.851083, 7.604854),
          (7.821661, 3.18878, 4.400032),

          (-7.640498, 0.756314, 4.967095),
          (-7.230136, 0.725585, 8.331517),
          (-5.748005, 0.480691, 11.193904),
          (-3.009834, 0.621885, 13.441012),
          (0.341982, 0.449246, 13.839247),
          (3.62126, 0.31676, 13.082255),
          (6.418348, 0.200262, 11.178412),
          (7.743287, 0.254288, 8.143276),
          (8.214926, 0.533799, 4.980188),

          (-7.794727, -1.924366, 4.686678),
          (-7.103159, -2.735806, 7.908936),
          (-5.549734, -3.131109, 10.995642),
          (-3.111164, -3.281632, 12.904391),
          (-0.072857, -3.405421, 13.509398),
          (3.044321, -3.820854, 12.781214),
          (5.712892, -3.643826, 10.907982),
          (7.304755, -3.111501, 7.913397),
          (7.92715, -2.443219, 4.673271),

          (-7.161848, -4.799244, 4.411572),
          (-6.375708, -5.683398, 7.142764),
          (-5.117089, -6.324777, 9.046002),
          (-2.8246, -6.605847, 10.717917),
          (-0.19569, -6.696784, 11.505725),
          (2.396374, -7.077637, 10.585553),
          (4.802065, -6.824497, 8.991351),
          (6.172683, -6.209247, 7.028114),
          (7.187716, -4.954237, 4.477674),

          (-5.894369, -6.974203, 4.318362),
          (-5.037746, -7.566237, 6.585544),
          (-2.544662, -8.415612, 7.820205),
          (-0.339835, -8.716856, 8.249729),
          (2.201964, -8.66148, 7.796194),
          (4.491326, -8.16103, 6.387415),
          (5.766648, -7.498684, 4.546538),

          (-6.387065, -5.755497, 1.886141),
          (-3.542601, -8.904578, 4.214279),
          (-0.080624, -9.660508, 4.670766),
          (3.050584, -9.25965, 4.194428),
          (6.192229, -6.797348, 2.355135),
          ]

    m1 = (m1 - np.min(m1)) / (np.max(m1) - np.min(m1))
    m1 = np.float32(np.array(m1))
    return m1



class CE_Label_Smooth_Loss(nn.Module):
    def __init__(self, classes=4, epsilon=0.14, ):
        super(CE_Label_Smooth_Loss, self).__init__()

        self.classes = classes
        self.epsilon = epsilon


    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * \
                 self.epsilon / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.epsilon))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss


def split_eye_data(eeg_eye_data, mode2_node_num):
    zero_index = []

    if mode2_node_num == 5: # MPED
        zero_index = [318, 319,
                      328, 329,
                      332, 333, 334, 335, 336, 337, 338, 339,
                      344, 345, 346, 347, 348, 349,
                      356, 357, 358, 359,
                      ]
    elif mode2_node_num == 6: # (6,6,4,4,4,9) SEED5
        zero_index =  [316, 317, 318, 319,
                       326, 327, 328, 329,
                       334, 335, 336, 337, 338, 339,
                       344, 345, 346, 347, 348, 349,
                       354, 355, 356, 357, 358, 359,
                       369
                       ]
    elif mode2_node_num == 7: # (8,8,2,4,2,2,2) MPED
        zero_index =  [318, 319,
                       328, 329,
                       332, 333, 334, 335, 336, 337, 338, 339,
                       344, 345, 346, 347, 348, 349,
                       352, 353, 354, 355, 356, 357, 358, 359,
                       362, 363, 364, 365, 366, 367, 368, 369,
                       372, 373, 374, 375, 376, 377, 378, 379,
                       ]
    elif mode2_node_num == 8: # (8,8,2,4,2,2,2) MPED
        zero_index =  [166,167,168,169,
                       176,177,178,179,
                       186,187,188,189,
                       196,197,198,199,
                       206,207,208,209,
                       216, 217, 218, 219,
                       226,227,228,229,
                       236, 237, 238, 239,
                       ]
    else:
        print("Wrong eye movement data arrangement")

    for i in range(len(zero_index)):
        eeg_eye_data = np.insert(eeg_eye_data, zero_index[i], 0, axis=1)


    return eeg_eye_data

DEAP_CHANNEL_NAME = [(-29.4367, 83.9171, -6.99),
                     (-33.7007, 76.8371, 21.227),
                     (-50.2438, 53.1112, 42.192),
                     (-70.2629, 42.4743, -11.42),
                     (-77.2149, 18.6433, 24.46),
                     (-34.0619, 26.0111, 79.987),
                     (-65.3581, -11.6317, 64.358),
                     (-84.1611, -16.0187, -9.346),
                     (-79.5922, -46.5507, 30.949),
                     (-35.5131, -47.2919, 91.315),
                     (-53.0073, -78.7878, 55.94),
                     (-72.4343, -73.4527, -2.487),
                     (-36.5114, -100.8529, 37.167),
                     (-29.4134, -112.449, 8.839),
                     (0.1076, -114.892, 14.657),
                     (0.3247, -81.115, 82.615),
                     (29.8723, 84.8959, -7.08),
                     (35.7123, 77.7259, 21.956),
                     (0.3122, 58.512, 66.462),
                     (51.8362, 54.3048, 40.814),
                     (73.0431, 44.4217, -12.0),
                     (79.5341, 19.9357, 24.438),
                     (34.7841, 26.4379, 78.808),
                     (0.4009, -9.167, 100.244),
                     (67.1179, -10.9003, 63.58),
                     (85.0799, -15.0203, -9.49),
                     (83.3218, -46.1013, 31.206),
                     (38.3838, -47.0731, 90.695),
                     (55.6667, -78.5602, 56.561),
                     (73.0557, -73.0683, -2.54),
                     (36.7816, -100.8491, 36.397),
                     (29.8426, -112.156, 8.8)]

def return_coordinates_deap():
    m1 = [(-29.4367, 83.9171, -6.99),
          (-33.7007, 76.8371, 21.227),
          (-50.2438, 53.1112, 42.192),
          (-70.2629, 42.4743, -11.42),
          (-77.2149, 18.6433, 24.46),
          (-34.0619, 26.0111, 79.987),
          (-65.3581, -11.6317, 64.358),
          (-84.1611, -16.0187, -9.346),
          (-79.5922, -46.5507, 30.949),
          (-35.5131, -47.2919, 91.315),
          (-53.0073, -78.7878, 55.94),
          (-72.4343, -73.4527, -2.487),
          (-36.5114, -100.8529, 37.167),
          (-29.4134, -112.449, 8.839),
          (0.1076, -114.892, 14.657),
          (0.3247, -81.115, 82.615),
          (29.8723, 84.8959, -7.08),
          (35.7123, 77.7259, 21.956),
          (0.3122, 58.512, 66.462),
          (51.8362, 54.3048, 40.814),
          (73.0431, 44.4217, -12.0),
          (79.5341, 19.9357, 24.438),
          (34.7841, 26.4379, 78.808),
          (0.4009, -9.167, 100.244),
          (67.1179, -10.9003, 63.58),
          (85.0799, -15.0203, -9.49),
          (83.3218, -46.1013, 31.206),
          (38.3838, -47.0731, 90.695),
          (55.6667, -78.5602, 56.561),
          (73.0557, -73.0683, -2.54),
          (36.7816, -100.8491, 36.397),
          (29.8426, -112.156, 8.8)
          ]

    m1 = (m1 - np.min(m1)) / (np.max(m1) - np.min(m1))
    m1 = np.float32(np.array(m1))
    return m1
