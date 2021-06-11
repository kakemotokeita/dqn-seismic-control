import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):

    def __init__(self, s, outputs, device = None):
        super(DQN, self).__init__()
        self.device = device if device else torch.device("cpu")
        self.conv1 = nn.Conv1d(1, 16, kernel_size=2, stride=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=2, stride=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 32, kernel_size=2, stride=1)
        self.bn3 = nn.BatchNorm1d(32)

        # conv1dから出力されるサイズの計算
        def conv1d_out_size(size, kernel_size=2, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1

        # conv1d3回分の出力サイズを計算
        conv = conv1d_out_size(conv1d_out_size(conv1d_out_size(s)))
        linear_input_size = conv * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # ネットワークの順伝播を計算して計算結果を返す
    def forward(self, x):
        x = x.to(self.device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))
