import torch
from torch.nn import Module, Sequential
from torch.nn import functional as F

from .layers import GBPerceptronVN, GBLinear


class FrontierLayerVN(Module):
    def __init__(self, in_sca, in_vec, hidden_dim_sca, hidden_dim_vec, bottleneck=1,
                 use_conv1d=False):
        super(FrontierLayerVN, self).__init__()
        self.net = Sequential(
            GBPerceptronVN(
                in_sca, in_vec, hidden_dim_sca, hidden_dim_vec, bottleneck=bottleneck,
                use_conv1d=use_conv1d
                ),
            GBLinear(
                hidden_dim_sca, hidden_dim_vec, 1, 1, bottleneck=bottleneck, use_conv1d=use_conv1d
                )
        )

    def forward(self, h_att, idx_ligans):
        h_att_ligand = [h_att[0][idx_ligans], h_att[1][idx_ligans]]
        pred = self.net(h_att_ligand)
        pred = pred[0]
        return pred

