import torch
from torch import nn
from torch.nn import Module, Sequential
from torch.nn import functional as F
from torch_scatter import scatter_add
from .layers import GDBPerceptronVN, GDBLinear, MessageModule, MessageAttention, AttentionEdges, ST_GDBP_Exp, VNLeakyReLU
from .net_utils import GaussianSmearing, EdgeExpansion
import math


GAUSSIAN_COEF = 1.0 / math.sqrt(2 * math.pi)
class PositionEncoder(Module):
    def __init__(self, in_sca, in_vec, edge_channels, num_filters, bottleneck=1, cutoff=10.,
                 num_heads=1, use_conv1d=False, with_root=True) -> None:
        super(PositionEncoder, self).__init__()
        self.message_module = MessageModule(
            in_sca, in_vec, edge_channels, edge_channels, num_filters[0], num_filters[1], 
            bottleneck, cutoff, use_conv1d=use_conv1d
            )
        self.message_att = MessageAttention(
            num_filters[0], num_filters[1], num_filters[0], num_filters[1], bottleneck=bottleneck, 
            num_heads=num_heads, use_conv1d=use_conv1d
            )
        self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=edge_channels)
        self.vector_expansion = EdgeExpansion(edge_channels)
        self.root_lin = GDBLinear(
            in_sca, in_vec, num_filters[0], num_filters[1], bottleneck=bottleneck, use_conv1d=use_conv1d
            )
        self.root_vector_expansion = EdgeExpansion(edge_channels)
    
    def forward(self, pos_query, edge_index_q_cps_knn, cpx_pos, node_attr_compose, atom_type_emb, 
                annealing=False):
        vec_ij = pos_query[edge_index_q_cps_knn[0]] - cpx_pos[edge_index_q_cps_knn[1]]
        dist_ij = torch.norm(vec_ij, p=2, dim=-1).view(-1, 1)  # (A, 1)
        edge_ij = self.distance_expansion(dist_ij), self.vector_expansion(vec_ij)

        #if isinstance(atom_type_emb, torch.Tensor) and self.with_root:
        root_vec_ij = self.root_vector_expansion(pos_query)
        y_root_sca, y_root_vec = self.root_lin([atom_type_emb, root_vec_ij])
        x = [y_root_sca, y_root_vec]

        # node_attr_ctx_j = [node_attr_ctx_[edge_index_q_cps_knn[1]] for node_attr_ctx_ in node_attr_ctx]  # (A, H)
        h_q = self.message_module(node_attr_compose, edge_ij, edge_index_q_cps_knn[1], dist_ij, annealing=annealing)
        y = self.message_att(x, h_q, edge_index_q_cps_knn[0])
        return y


class BondFlow(Module):
    def __init__(self, in_sca, in_vec, edge_channels, num_filters, num_bond_types=3, num_heads=4,
                 cutoff=10.0, with_root=True, num_st_layers=6, bottleneck=1, use_conv1d=False):
        super(BondFlow, self).__init__()
        self.with_root = with_root
        self.num_bond_types = num_bond_types
        self.num_st_layers = num_st_layers
        ## query encoder
        self.pos_encoder = PositionEncoder(
            in_sca, in_vec, edge_channels, num_filters, cutoff=cutoff, bottleneck=bottleneck,
            use_conv1d=use_conv1d
            )
        ## edge pred
        self.distance_expansion_3A = GaussianSmearing(stop=3., num_gaussians=edge_channels)
        self.vector_expansion = EdgeExpansion(edge_channels)
        self.nn_edge_ij = Sequential(
            GDBPerceptronVN(edge_channels, edge_channels, num_filters[0], num_filters[1]),
            GDBLinear(num_filters[0], num_filters[1], num_filters[0], num_filters[1])
        )
        self.edge_feat = Sequential(
            GDBPerceptronVN(num_filters[0] * 2 + in_sca, num_filters[1] * 2 + in_vec, num_filters[0], num_filters[1]),
            GDBLinear(num_filters[0], num_filters[1], num_filters[0], num_filters[1])
        )
        self.edge_atten = AttentionEdges(num_filters, num_filters, num_heads, num_bond_types)
        ## flow layer
        self.flow_layers = torch.nn.ModuleList()
        for _ in range(num_st_layers):
            flow_layer = ST_GDBP_Exp(
                num_filters[0], 
                num_filters[1], 
                num_bond_types + 1, 
                num_filters[1],
                bottleneck=bottleneck, 
                use_conv1d=use_conv1d
                )
            self.flow_layers.append(flow_layer)
    
    def forward(self, z_edge, pos_query, edge_index_query, cpx_pos, node_attr_compose, edge_index_q_cps_knn,
                atom_type_emb, index_real_cps_edge_for_atten=[], tri_edge_index=[], tri_edge_feat=[], 
                annealing=False):
        y = self.pos_encoder(
                    pos_query, edge_index_q_cps_knn, cpx_pos, node_attr_compose,
                    atom_type_emb, annealing=annealing
                    )
        if (len(edge_index_query) != 0) and (edge_index_query.size(1) > 0):
            idx_node_i = edge_index_query[0]
            node_mol_i = [
                y[0][idx_node_i],
                y[1][idx_node_i]
            ]
            idx_node_j = edge_index_query[1]
            node_mol_j = [
                node_attr_compose[0][idx_node_j],
                node_attr_compose[1][idx_node_j]
            ]
            vec_ij = pos_query[idx_node_i] - cpx_pos[idx_node_j]
            dist_ij = torch.norm(vec_ij, p=2, dim=-1).view(-1, 1)  # (E, 1)

            edge_ij = self.distance_expansion_3A(dist_ij), self.vector_expansion(vec_ij) 
            edge_feat = self.nn_edge_ij(edge_ij)  # (E, F)

            edge_attr = (torch.cat([node_mol_i[0], node_mol_j[0], edge_feat[0]], dim=-1),  # (E, F)
                         torch.cat([node_mol_i[1], node_mol_j[1], edge_feat[1]], dim=1))
            edge_attr = self.edge_feat(edge_attr)
            edge_attr = self.edge_atten(
                edge_attr, edge_index_query, cpx_pos, index_real_cps_edge_for_atten, 
                tri_edge_index, tri_edge_feat
                )
            #self.edge_atten()
            for ix in range(len(self.flow_layers)):
                s, t = self.flow_layers[ix](edge_attr)
                s = s.exp()
                z_edge = (z_edge + t) * s
                if ix == 0:
                    edge_log_jacob = (torch.abs(s) + 1e-20).log()
                else:
                    edge_log_jacob += (torch.abs(s) + 1e-20).log()
            return z_edge, edge_log_jacob
        else:
            z_edge = torch.empty([0, self.num_bond_types+1], device=pos_query.device)
            edge_log_jacob = torch.empty([0, self.num_bond_types+1], device=pos_query.device)
            return z_edge, edge_log_jacob

    def reverse(self, edge_latent, pos_query, edge_index_query, cpx_pos, node_attr_compose, edge_index_q_cps_knn,
                atom_type_emb, index_real_cps_edge_for_atten=[], tri_edge_index=[], tri_edge_feat=[], 
                annealing=False):
        y = self.pos_encoder(
                pos_query, edge_index_q_cps_knn, cpx_pos, node_attr_compose,
                atom_type_emb, annealing=annealing
                )
        if (len(edge_index_query) != 0) and (edge_index_query.size(1) > 0):
            idx_node_i = edge_index_query[0]
            node_mol_i = [
                y[0][idx_node_i],
                y[1][idx_node_i]
            ]
            idx_node_j = edge_index_query[1]
            node_mol_j = [
                node_attr_compose[0][idx_node_j],
                node_attr_compose[1][idx_node_j]
            ]
            vec_ij = pos_query[idx_node_i] - cpx_pos[idx_node_j]
            dist_ij = torch.norm(vec_ij, p=2, dim=-1).view(-1, 1)  # (E, 1)

            edge_ij = self.distance_expansion_3A(dist_ij), self.vector_expansion(vec_ij) 
            edge_feat = self.nn_edge_ij(edge_ij)  # (E, F)

            edge_attr = (torch.cat([node_mol_i[0], node_mol_j[0], edge_feat[0]], dim=-1),  # (E, F)
                         torch.cat([node_mol_i[1], node_mol_j[1], edge_feat[1]], dim=1))
            edge_attr = self.edge_feat(edge_attr)
            edge_attr = self.edge_atten(
                edge_attr, edge_index_query, cpx_pos, index_real_cps_edge_for_atten, 
                tri_edge_index, tri_edge_feat
                )
            #self.edge_atten()
            for ix in range(len(self.flow_layers)):
                s, t = self.flow_layers[ix](edge_attr)
                edge_latent = (edge_latent / s.exp()) - t
            return edge_latent
        else:
            edge_latent = torch.empty([0, self.num_bond_types+1], device=pos_query.device)
            return edge_latent
