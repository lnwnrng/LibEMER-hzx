import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from itertools import chain

class BDDAE(nn.Module):
    def __init__(self, eeg_dim, bio_dim,h_eeg_dim, h_bio_dim, joint_dim, eeg_dropout_rate, bio_dropout_rate,num_classes):
        super(BDDAE, self).__init__()
        self.h_eeg_dim = h_eeg_dim
        self.h_bio_dim = h_bio_dim

        self.eeg_corruption = nn.Dropout(p=eeg_dropout_rate)
        self.bio_corruption = nn.Dropout(p=bio_dropout_rate)

        # ---- 编码器部分 ----
        # 模态1 (EEG) 的编码器
        self.encoder_eeg = nn.Linear(eeg_dim, h_eeg_dim)
        # 模态2 (Eye) 的编码器
        self.encoder_bio = nn.Linear(bio_dim, h_bio_dim)
        # 联合编码器，输入为拼接后的向量
        self.encoder_joint = nn.Linear(h_eeg_dim + h_bio_dim, joint_dim)

        # ---- 解码器部分 (与编码器对称) ----
        # 联合解码器
        self.decoder_joint_bias = nn.Parameter(torch.randn(h_eeg_dim + h_bio_dim))

        self.decoder_eeg_bias = nn.Parameter(torch.randn(eeg_dim))
        
        self.decoder_bio_bias = nn.Parameter(torch.randn(bio_dim))

        self.sigmoid = nn.Sigmoid()

        self.classifer = nn.Linear(joint_dim, num_classes)

    def forward(self, eeg_feature, bio_feature):
        eeg_feature = eeg_feature.reshape(eeg_feature.shape[0], -1)
        bio_feature = bio_feature.reshape(bio_feature.shape[0], -1)
        
        eeg_feature = self.eeg_corruption(eeg_feature)
        bio_feature = self.bio_corruption(bio_feature)

        h_eeg = self.sigmoid(self.encoder_eeg(eeg_feature))
        h_bio = self.sigmoid(self.encoder_bio(bio_feature))

        h_joint = self.sigmoid(self.encoder_joint(torch.cat((h_eeg, h_bio), dim=1)))

        d_joint = self.sigmoid(F.linear(h_joint, self.encoder_joint.weight.T, self.decoder_joint_bias))
        
        # 将解码后的联合表示重新划为两个模态
        d_h_eeg, d_h_eye = torch.split(d_joint, [self.h_eeg_dim, self.h_bio_dim], dim=1)
        
        # 重建原始输入
        recon_eeg = self.sigmoid(F.linear(d_h_eeg, self.encoder_eeg.weight.T, self.decoder_eeg_bias))
        recon_bio = self.sigmoid(F.linear(d_h_eye, self.encoder_bio.weight.T, self.decoder_bio_bias))

        prediction = self.classifer(h_joint)

        return recon_eeg, recon_bio, prediction

    def encoder_parameters(self):
        """返回所有参与重构任务的参数"""
        return chain(
            self.encoder_eeg.parameters(),
            self.encoder_bio.parameters(),
            self.encoder_joint.parameters(),
            [self.decoder_joint_bias, self.decoder_eeg_bias, self.decoder_bio_bias]
        )

    # 新增：辅助方法，返回所有分类器相关的参数
    def classifier_parameters(self):
        """返回所有参与分类任务的参数"""
        return self.classifer.parameters()