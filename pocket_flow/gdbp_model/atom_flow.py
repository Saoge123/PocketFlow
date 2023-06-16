import torch
from torch import nn
from torch.nn import functional as F
from .layers import GDBPerceptronVN, GDBLinear, ST_GDBP_Exp



class AtomFlow(nn.Module):
    def __init__(self, in_sca, in_vec, hidden_dim_sca, hidden_dim_vec, num_lig_atom_type=10,
                 num_flow_layers=6, bottleneck=1, use_conv1d=False) -> None:
        super(AtomFlow, self).__init__()
        
        self.net = nn.Sequential(
            GDBPerceptronVN(
                in_sca, in_vec, hidden_dim_sca, hidden_dim_vec, bottleneck=bottleneck, use_conv1d=use_conv1d
                ),
            GDBLinear(
                hidden_dim_sca, hidden_dim_vec, hidden_dim_sca, hidden_dim_vec, bottleneck=bottleneck, 
                use_conv1d=use_conv1d
                )
        )
        
        self.flow_layers = nn.ModuleList()
        for _ in enumerate(range(num_flow_layers)):
            layer = ST_GDBP_Exp(
                hidden_dim_sca, hidden_dim_vec, num_lig_atom_type, hidden_dim_vec, bottleneck=bottleneck,
                use_conv1d=use_conv1d
            )
            self.flow_layers.append(layer)

    def forward(self, z_atom, compose_features, focal_idx):
        sca_focal, vec_focal = compose_features[0][focal_idx], compose_features[1][focal_idx]
        sca_focal, vec_focal = self.net([sca_focal, vec_focal])
        for ix in range(len(self.flow_layers)):
            s, t = self.flow_layers[ix]([sca_focal, vec_focal])
            s = s.exp()
            z_atom = (z_atom + t) * s
            if ix == 0:
                atom_log_jacob = (torch.abs(s) + 1e-20).log()
            else:
                atom_log_jacob += (torch.abs(s) + 1e-20).log()
        return z_atom, atom_log_jacob

    def reverse(self, atom_latent, compose_features, focal_idx):
        sca_focal, vec_focal = compose_features[0][focal_idx], compose_features[1][focal_idx]
        sca_focal, vec_focal = self.net([sca_focal, vec_focal])
        for ix in range(len(self.flow_layers)):
            s, t = self.flow_layers[ix]([sca_focal, vec_focal])
            atom_latent = (atom_latent / s.exp()) - t
        return atom_latent