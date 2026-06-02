"""
CNN-BiLSTM (CRNN) 1-D segmenter — the other dominant PCG-segmentation architecture, used to
show the deep-fails-under-in-band-noise finding and the hybrid rescue are NOT specific to the
U-Net. Same interface as UNet1D: forward(x[B,1,T]) -> logits[B,n_classes,T].
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CRNN1D(nn.Module):
    def __init__(self, in_ch=1, n_classes=5, base=32, hidden=64, lstm_layers=2):
        super().__init__()
        def cb(i, o):
            return nn.Sequential(nn.Conv1d(i, o, 7, stride=2, padding=3),
                                 nn.BatchNorm1d(o), nn.ReLU())
        self.enc = nn.Sequential(cb(in_ch, base), cb(base, base * 2), cb(base * 2, base * 2))  # /8
        self.lstm = nn.LSTM(base * 2, hidden, num_layers=lstm_layers,
                            batch_first=True, bidirectional=True)
        self.head = nn.Conv1d(2 * hidden, n_classes, 1)

    def forward(self, x):
        T = x.shape[-1]
        h = self.enc(x)                 # [B, C, T/8]
        h = h.transpose(1, 2)           # [B, T/8, C]
        h, _ = self.lstm(h)             # [B, T/8, 2*hidden]
        h = h.transpose(1, 2)           # [B, 2*hidden, T/8]
        h = F.interpolate(h, size=T, mode="linear", align_corners=False)
        return self.head(h)             # [B, n_classes, T]
