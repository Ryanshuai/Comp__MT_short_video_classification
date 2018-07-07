import torch
import torch.nn as nn
import torch.tensor as tensor
import torch.nn.functional as F
import math


class ContextGating(nn.Module):
    def __init__(self,
                 input_size: int,
                 remove_diag: bool,
                 add_BN: bool,
                 ):
        super().__init__()
        self.remove_diag = remove_diag
        self.add_BN = add_BN
        if add_BN:
            self.bn = nn.BatchNorm1d(input_size)
            self.fc = nn.Linear(input_size, input_size, bias=False)
        else:
            self.fc = nn.Linear(input_size, input_size)

    def forward(self,
                net_in: tensor
                ):
        gates = self.fc(net_in)

        if self.remove_diag:
            gating_weights = list(self.fc.parameters())[0]
            diagonals = gating_weights.diag()
            gates = gates - torch.mul(net_in, diagonals)

        if self.add_BN:
            gates = self.bn(gates)

        gates = F.sigmoid(gates)
        net_out = torch.mul(net_in, gates)
        return net_out


class NetVLAD(nn.Module):
    def __init__(self,
                 k: int,  # k means center number
                 d: int,  # feature dim
                 add_BN: bool,
                 ):
        super().__init__()
        self.k = k
        self.d = d
        self.add_BN = add_BN
        if add_BN:
            self.a_BN = nn.BatchNorm1d(k)
            self.fc = nn.Linear(d, k, bias=False)
        else:
            self.fc = nn.Linear(d, k)
        stddev = 1 / math.sqrt(d)
        self.k_mean = torch.randn(k, d, requires_grad=True)*stddev
        self.softmax = nn.Softmax(dim=2)

    def forward(self,
                net_in: tensor,                            # shape == [bs, n, d]
                ):
        x = net_in
        n = x.shape()[1]
        activation = self.fc(x)                            # shape == [bs, n, k]
        if self.add_BN:
            activation = activation.view(-1, self.k)       # shape == [bs, n, k]
            activation = self.a_BN(activation)             # shape == [bs*n, k]
            activation = activation.view(-1, n, self.k)    # shape == [bs, n, k]
        activation = self.softmax(activation)              # shape == [bs, n, k]
        activation = activation.unsqueeze(dim=3)           # shape == [bs, n, k, 1]

        x = x.unsqueeze(dim=2)                             # shape == [bs, n, 1, d]
        k_mean = self.k_mean.unsqueeze(dim=0)              # shape == [1, k, d]
        k_mean = k_mean.unsqueeze(dim=0)                   # shape == [1, 1, k, d]
        core = x - k_mean                                  # shape == [bs, n, k, d]

        x_sub_c = torch.mul(activation, core)              # shape == [bs, n, k, d]
        v = torch.sum(x_sub_c, 1)                          # shape == [bs, k, d]
        v = F.normalize(v, dim=2)                          # shape == [bs, k, d]
        v = v.view(v.size(0), -1)                          # shape == [bs, k*d]
        v = F.normalize(v, dim=1)                          # shape == [bs, k*d]

        return v