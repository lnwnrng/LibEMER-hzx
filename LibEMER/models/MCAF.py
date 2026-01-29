import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ChannelAttention(nn.Module):
    def __init__(self, channels, seq_len, d_model, nhead):
        super().__init__()
        self.channels = channels
        self.seq_len = seq_len
        self.d_model = d_model
        self.nhead = nhead
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=nhead, batch_first=True)
    def forward(self, x):
        # x: [B, C, T]
        x_v = x.transpose(1, 2)  # [B, T, C]
        pooled = self.avgpool(x)  # [B, C, 1]
        pooled = pooled.squeeze(-1)  # [B, C]
        pooled = pooled.unsqueeze(1)  # [B, 1, C]
        Q = pooled.expand(-1, x_v.shape[1], -1)  # [B, T, C]
        K = pooled.expand(-1, x_v.shape[1], -1)  # [B, T, C]
        V = x_v  # [B, T, C]
        attn_out, _ = self.attn(Q, K, V)
        x_weighted = x_v * attn_out
        return x_weighted.transpose(1, 2)  # [B, C, T]

class SpatialCNN(nn.Module):
    def __init__(self, in_channels, seq_len, out_features):
        super().__init__()
        self.conv = nn.Conv2d(1, 32, kernel_size=(in_channels, seq_len))
        self.elu = nn.ELU()
        self.pool = nn.MaxPool2d(kernel_size=(1, 2))
        self.fc = nn.Linear(32, out_features)
    def forward(self, x):
        x = x.unsqueeze(1)  # [B, 1, C, T]
        x = self.conv(x)    # [B, 32, 1, 1] or [B, 32, 1, W]
        x = self.elu(x)
        if x.shape[-1] > 1:
            x = self.pool(x)
        x = x.view(x.size(0), -1)  # [B, 32]
        x = self.fc(x)      # [B, out_features]
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    def forward(self, x):
        # x: [B, T, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return x

class MCAF(nn.Module):
    def __init__(self, num_classes=3, eeg_channels=62, eog_channels=1, seq_len_eeg=5, seq_len_eog=33, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        # 自动选择eeg_channels的最大因数作为nhead
        def get_max_divisor(n, max_head=8):
            return max([h for h in range(1, max_head+1) if n % h == 0])
        eeg_nhead = get_max_divisor(eeg_channels, max_head=8)
        eog_nhead = get_max_divisor(eog_channels, max_head=1)
        # 通道注意力
        self.eeg_channel_attn = ChannelAttention(eeg_channels, seq_len_eeg, d_model, eeg_nhead)
        self.eog_channel_attn = ChannelAttention(eog_channels, seq_len_eog, d_model, eog_nhead)
        # 空间特征
        self.eeg_spatial = SpatialCNN(eeg_channels, seq_len_eeg, d_model)
        self.eog_spatial = SpatialCNN(eog_channels, seq_len_eog, d_model)
        # 融合
        self.fusion_fc = nn.Linear(d_model*2, d_model)
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len=64)
        # Transformer分类器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.7,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)
    def forward(self, eeg, eog):
        # eeg: [B, S, C, T] or [B, C, T]，eog: [B, S, 1, T] or [B, 1, T]
        if eeg.dim() == 4:
            eeg = eeg[:,0]  # 取第一个session
        if eog.dim() == 4:
            eog = eog[:,0]
        # 通道注意力
        eeg = self.eeg_channel_attn(eeg)  # [B, C, T]
        eog = self.eog_channel_attn(eog)  # [B, 1, T]
        # 空间特征
        eeg_feat = self.eeg_spatial(eeg)  # [B, d_model]
        eog_feat = self.eog_spatial(eog)  # [B, d_model]
        # 融合
        feat = torch.cat([eeg_feat, eog_feat], dim=-1)  # [B, d_model*2]
        feat = self.fusion_fc(feat)  # [B, d_model]
        # 变为序列长度1
        feat = feat.unsqueeze(1)  # [B, 1, d_model]
        # 位置编码
        feat = self.pos_encoder(feat)
        # Transformer
        feat = self.transformer(feat)  # [B, 1, d_model]
        feat = self.norm(feat)
        # 分类
        out = self.classifier(feat[:,0])  # [B, num_classes]
        return out