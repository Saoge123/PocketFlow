import torch
from torch import nn
from .layers import AttentionInteractionBlockVN



class ContextEncoder(nn.Module):
    
    def __init__(self, hidden_channels=[256, 64], edge_channels=64, num_edge_types=4, 
                 key_channels=128, num_heads=4, num_interactions=6, k=32, cutoff=10.0,
                 bottleneck=1, use_conv1d=False):
        super(ContextEncoder, self).__init__()
        self.hidden_channels = hidden_channels
        self.edge_channels = edge_channels
        self.key_channels = key_channels  # not use
        self.num_heads = num_heads  # not use
        self.num_interactions = num_interactions
        self.k = k
        self.cutoff = cutoff

        self.interactions = nn.ModuleList()
        for _ in range(num_interactions):
            block = AttentionInteractionBlockVN(
                hidden_channels=hidden_channels,
                edge_channels=edge_channels,
                num_edge_types=num_edge_types,
                num_heads=num_heads,
                cutoff = cutoff,
                bottleneck=bottleneck,
                use_conv1d = use_conv1d
            )
            self.interactions.append(block)

    @property
    def out_sca(self):
        return self.hidden_channels[0]
    
    @property
    def out_vec(self):
        return self.hidden_channels[1]

    def forward(self, node_attr, pos, edge_index, edge_feature, annealing=True):

        edge_vector = pos[edge_index[0]] - pos[edge_index[1]]
        edge_dist = torch.norm(edge_vector, dim=-1, p=2)
        h = list(node_attr)
        for interaction in self.interactions:
            delta_h = interaction(h, edge_index, edge_feature, edge_vector, edge_dist, annealing=annealing)
            h[0] = h[0] + delta_h[0]
            h[1] = h[1] + delta_h[1]
        return h