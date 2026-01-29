import os
import numpy as np
import pickle
import torch
import torch.nn as nn

class BimodalLSTM(nn.Module):
    def __init__(self, eeg_feature_dim, bio_feature_dim, eeg_hidden_size, bio_hidden_size, num_layers, eeg_dropout_rate,bio_dropout_rate,num_classes):
        super(BimodalLSTM, self).__init__()
        self.eeg_feature_dim = eeg_feature_dim
        self.bio_feature_dim = bio_feature_dim
        self.eeg_hidden_size = eeg_hidden_size
        self.bio_hidden_size = bio_hidden_size
        self.num_layers = num_layers
        self.eeg_dropout_rate = eeg_dropout_rate
        self.bio_drop_rate = bio_dropout_rate
        self.num_classes = num_classes
        
        self.eeglstm = nn.LSTM(input_size=eeg_feature_dim, hidden_size=eeg_hidden_size, num_layers=num_layers, batch_first=True, 
                               dropout=eeg_dropout_rate if num_layers > 1 else 0, bidirectional= False)
        
        self.biolstm = nn.LSTM(input_size=bio_feature_dim, hidden_size=bio_hidden_size, num_layers=num_layers, batch_first=True, 
                               dropout=bio_dropout_rate if num_layers > 1 else 0, bidirectional= False)
        
        self.eeg_dropout = nn.Dropout(eeg_dropout_rate)
        self.bio_dropout = nn.Dropout(bio_dropout_rate)

        self.classifer = nn.Linear(eeg_hidden_size + bio_hidden_size, num_classes)

    def forward(self, eeg, bio):
        eeg = eeg.reshape(eeg.shape[0], eeg.shape[1], -1)
        bio = bio.reshape(bio.shape[0], bio.shape[1], -1)
        eeg_lstm_out, (h_n_eeg, c_n_eeg) = self.eeglstm(eeg)
        bio_lstm_out, (h_n_bio, c_n_bio) = self.biolstm(bio)

        eeg_features = eeg_lstm_out[:, -1, :] 
        bio_features = bio_lstm_out[:, -1, :]

        eeg_features = self.eeg_dropout(eeg_features)
        bio_features = self.bio_dropout(bio_features)

        fused_features = torch.cat((eeg_features, bio_features), dim=1)
        output = self.classifer(fused_features)

        return output
        


 