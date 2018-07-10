import torch
import torch.nn as nn
import torch.tensor as tensor
import torch.nn.functional as F
import math


class ContextGating(nn.Module):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 remove_diag: bool,
                 add_BN: bool,
                 ):
        super().__init__()
        self.remove_diag = remove_diag
        self.add_BN = add_BN
        if add_BN:
            self.bn = nn.BatchNorm1d(input_size)
            self.fc = nn.Linear(input_size, output_size, bias=False)
        else:
            self.fc = nn.Linear(input_size, output_size)
            nn.init.xavier_normal(self.fc.bias)
        nn.init.xavier_normal(self.fc.weight)

    def forward(self,
                net_in: tensor,
                gate_in: tensor, # gate_in can be same with net_in
                ):
        gates = self.fc(gate_in)

        if self.remove_diag:
            gating_weights = self.fc.weight
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
            nn.init.xavier_normal(self.fc.bias)
        nn.init.xavier_normal(self.fc.weight)
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


class MoE(nn.Module):
    def __init__(self,
                 feature_size: int,
                 num_classes: int,
                 num_mixture: int, # default is 2
                 low_rank_gating: int, # default is -1
                 gating_prob: bool,
                 gating_input_prob: bool,
                 remove_diag: bool
                 ):
        super().__init__()
        self.num_mixture = num_mixture
        self.num_class = num_classes
        self.low_rank_gating = low_rank_gating
        self.gating_prob = gating_prob
        self.gating_input_prob = gating_input_prob

        # gating of expert
        output_size = num_classes * (num_mixture + 1)
        if low_rank_gating == -1:
            self.gating_activation_fc = nn.Linear(feature_size, output_size) # ??? the weight needs to be penalty by l2_regularizer
        else:
            self.gating_activation_fc1 = nn.Linear(feature_size, low_rank_gating) # the weight needs to be penalty by l2_regularizer
            self.gating_activation_fc = nn.Linear(low_rank_gating, output_size) # the weight needs to be penalty by l2_regularizer
        self.gating_distributionn_softmax = nn.Softmax()

        # experts
        self.expert_activation = nn.Linear(feature_size, num_classes * num_mixture) # the weight needs to be penalty by l2_regularizer
        self.expert_distrubition_sigmoid = nn.Sigmoid()

        if gating_prob:
            if gating_input_prob == 'prob':
                self.context_gating = ContextGating(num_classes, num_classes,remove_diag, True)
            else:
                self.context_gating = ContextGating(feature_size, num_classes, remove_diag, True)


    def forward(self,
                net_in: tensor,                            # shape == [bs, d]
                ):

        # gating of expert
        if self.low_rank_gating == -1:
            gate_activations = self.gating_activation_fc(net_in)
        else:
            g = self.gating_activation_fc1(net_in)
            gate_activations = self.gating_activation_fc(g)
            # [bs, num_class *(num_mixture+1)]
        gatint_distribution = self.gating_distributionn_softmax(gate_activations.view(-1, self.num_mixture+1))  #[bs * num_class , num_mixture+1]

        # expert
        expert_activation = self.expert_activation(net_in)  # [bs, num_class * num_mixture]
        expert_distribution = self.expert_distrubition_sigmoid(expert_activation.view(-1, self.num_mixture)) # [bs * num_class, num_mixture]

        # prob
        prob_by_class_and_batch = nn.sum(
                    gatint_distribution[:, :self.num_mixtures] * expert_distribution, 1) # [bs*num_class,1]

        prob = prob_by_class_and_batch.view(-1,self.num_class) # [bs, num_class]

        if self.gating_prob:
            if self.gating_input_prob == 'prob':
                probabilities = self.context_gating.forward(net_in=prob, gate_in=prob)
            else:
                probabilities = self.context_gating.forward(net_in=prob, gate_in=net_in) # [bs, num_class]
                
        return probabilities
