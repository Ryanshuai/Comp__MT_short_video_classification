import torch
import torch.nn as nn
import torch.tensor as tensor

from models.layers import NetVLAD, ContextGating, MoE

# Setting
RELU = True
Gating_for_NetVLAD = False # default is False in the code
Gating_Remove_Diag = False # default is False in the code

# MoE Default Setting in the Original Code
Num_Mixture = 2
Low_Rank_Gating = -1
Gating_Prob = False
Prob_Gating_Input = 'prob'
Remove_Diag = False # have no default value ???

class NetVLAD_FC_GATE(nn.Module):

    def __init__(self,
                 num_classes:int,
                 cluster_size: int,
                 video_dim: int,
                 audio_dim: int,
                 num_frames: int,
                 add_bn: bool,
                 hidden_size: int,
                 ):
        self.num_frames = num_frames
        self.video_dim = video_dim
        self.audio_dim = audio_dim
        self.add_bn = add_bn

        # input bn
        if add_bn:
            self.input_bn = nn.BatchNorm1d(video_dim+audio_dim)

        #   vlad
        self.video_vlad = NetVLAD(cluster_size, video_dim, add_bn)
        self.audio_vlad = NetVLAD(cluster_size / 2, audio_dim, add_bn)

        # fc
        if add_bn or RELU:
            self.fc = nn.Linear(cluster_size * video_dim + cluster_size/2 * audio_dim, hidden_size, bias=False)
            self.fc_bn = nn.BatchNorm1d(hidden_size)
        else:
            self.fc = nn.Linear(cluster_size * video_dim + cluster_size/2 * audio_dim, hidden_size)
        if RELU:
            self.relu6 = nn.ReLU6()

        # Context Gating
        if Gating_for_NetVLAD:
            self.gating_vlad = ContextGating(hidden_size,hidden_size, Gating_Remove_Diag, self.add_bn)

        # MoE
        self.moe = MoE(hidden_size, num_classes, Num_Mixture, Low_Rank_Gating, Gating_Prob, Prob_Gating_Input, Remove_Diag)
    def forward(self,
                net_in: tensor, #[bs, n, video_dim + audio_dim]
                ):
        # input
        reshaped_input = net_in.view(-1, self.video_dim + self.audio_vlad) # [bs * n, video_dim + audio_dim ]
        if self.add_bn:
            reshaped_input = self.input_bn(reshaped_input)

        # vlad
        video_input = reshaped_input[:, 0:self.video_dim].view(-1, self.num_frames, self.video_dim)
        audio_input = reshaped_input[:, self.video_dim:].view(-1, self.num_frames, self.audio_dim)
        video_vlad = self.video_vlad.forward(video_input) #[bs, cluster_size * video_dim]
        audio_vlad = self.audio_vlad.forward(audio_input) #[bs, cluster_size/2 * audio_dim]
        vlad = torch.cat([video_vlad, audio_vlad], 1)  #[bs, cluster_size * video_dim + cluster_size/2 * audio_dim]

        # fc
        if  self.add_bn or RELU:
            activation = self.fc(vlad)
            activation = self.fc_bn(activation)
        else:
            activation = self.fc(vlad) # [bs, hidden_size]

        if RELU:
            activation = self.relu6(activation)

        if Gating_for_NetVLAD:
            activation = self.gating_vlad.forward(activation,activation)  # [bs, hidden_size]

        probabilities = self.moe.forward(activation) #[bs, num_classes]

