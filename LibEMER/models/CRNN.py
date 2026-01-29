import torch
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, eeg_feature_dim, bio_feature_dim, bio_channels, num_classes=2):
        super(CRNN, self).__init__()
        self.eeg_feature_dim = eeg_feature_dim
        self.bio_feature_dim = bio_feature_dim
        self.bio_channels = bio_channels

        # EEG分支
        self.eeg_conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.eeg_conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.eeg_conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.eeg_final_conv = nn.Conv2d(256, 15, kernel_size=1)
        
        # PERI分支
        self.peri_fc = nn.Linear(bio_channels, 64)
        self.peri_lstm1 = nn.LSTM(64, 128, batch_first=True)
        self.peri_lstm2 = nn.LSTM(128, 128, batch_first=True)
        
        # 分类器
        self.classifier = nn.Linear(15 * 9 * 9 + 128, num_classes)
        
    def forward(self, eeg, peri):
        # EEG处理
        eeg = eeg.permute(0, 3, 1, 2).unsqueeze(2)  # batch*feature*1*9*9
        batch_size = eeg.size(0)

        cnn_outputs = []
        for t in range(self.eeg_feature_dim):
            x = self.eeg_conv1(eeg[:, t, :, :, :])
            x = self.eeg_conv2(x)
            x = self.eeg_conv3(x)
            cnn_outputs.append(x)
        
        x = torch.stack(cnn_outputs, dim=1)
        x = x.view(batch_size * self.eeg_feature_dim, 256, 9, 9)
        x = self.eeg_final_conv(x)
        x = x.view(batch_size, self.eeg_feature_dim, 15, 9, 9)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(batch_size, self.eeg_feature_dim, -1)
        eeg_features = torch.mean(x, dim=1)
        
        # PERI处理
        peri = peri.permute(0, 2, 1)  # [batch, seq_len, 8]
        peri = self.peri_fc(peri)
        lstm1_out, _ = self.peri_lstm1(peri)
        lstm2_out, _ = self.peri_lstm2(lstm1_out)
        peri_features = lstm2_out[:, -1, :]  # 取最后一个时间步
        
        # 分类
        combined = torch.cat((eeg_features, peri_features), dim=1)
        return self.classifier(combined)