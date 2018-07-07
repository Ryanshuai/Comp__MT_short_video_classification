import torch
import torch.nn as nn
import torch.tensor as tensor
import torch.nn.functional as F


class NetVLAD(nn.Module):
    def __init__(self, k: int,  # k means center number
                 d: int,  # feature dim
                 add_BN: bool,
                 ):
        super().__init__()
        self.add_BN = add_BN
        if self.add_BN:
            belong_BN = nn.BatchNorm1d()
        self.fc = nn.Linear(d, k)
        self.k_mean = torch.rand(k, d, requires_grad=True)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, net_in: tensor,  # shape == [bs, n, d]
                ):
        x = net_in
        belong = self.fc(x)                     # shape == [bs, n, k]
        if self.add_BN:
            belong =
        belong = self.softmax(belong)           # shape == [bs, n, k]
        belong = belong.unsqueeze(dim=3)        # shape == [bs, n, k, 1]

        x = x.unsqueeze(dim=2)                  # shape == [bs, n, 1, d]
        k_mean = self.k_mean.unsqueeze(dim=0)   # shape == [1, k, d]
        k_mean = k_mean.unsqueeze(dim=0)        # shape == [1, 1, k, d]
        core = x - k_mean                       # shape == [bs, n, k, d]

        x_sub_c = belong*core                   # shape == [bs, n, k, d]
        v = torch.sum(x_sub_c, 1)               # shape == [bs, k, d]
        v = F.normalize(v, dim=2)               # shape == [bs, k, d]
        v = v.view(v.size(0), -1)               # shape == [bs, k*d]
        v = F.normalize(v, dim=2)               # shape == [bs, k*d]

        return v