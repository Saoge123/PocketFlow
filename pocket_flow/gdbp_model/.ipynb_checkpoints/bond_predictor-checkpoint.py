import torch
from torch import nn
from torch.nn import Module, Sequential
from torch.nn import functional as F
from torch_scatter import scatter_add
#from .pos_filter import PositionEncoder
from .layers import GBPerceptronVN, GBLinear, MessageModule, MessageAttention, AttentionEdges, ST_GBP_Exp, VNLeakyReLU
from .net_utils import GaussianSmearing, EdgeExpansion
import math


GAUSSIAN_COEF = 1.0 / math.sqrt(2 * math.pi)

class BondPredictor(Module):
    def __init__(self, in_sca, in_vec, edge_channels, num_filters, num_bond_types, 
                 num_heads=4, cutoff=10.0, with_root=True, bottleneck=1):
        super(BondPredictor, self).__init__()
        self.with_root = with_root
        self.num_bond_types = num_bond_types
        self.message_module = MessageModule(
            in_sca, in_vec, edge_channels, edge_channels, num_filters[0], num_filters[1], cutoff=cutoff,
            bottleneck=bottleneck
            )
        self.nn_edge_ij = Sequential(
            GBPerceptronVN(edge_channels, edge_channels, num_filters[0], num_filters[1], bottleneck=bottleneck),
            GBLinear(num_filters[0], num_filters[1], num_filters[0], num_filters[1], bottleneck=bottleneck)
        )

        self.edge_feat = Sequential(
            GBPerceptronVN(num_filters[0] * 2 + in_sca, num_filters[1] * 2 + in_vec, num_filters[0], num_filters[1],
                    bottleneck=bottleneck),
            GBLinear(num_filters[0], num_filters[1], num_filters[0], num_filters[1], bottleneck=bottleneck)
        )
        self.edge_atten = AttentionEdges(
            num_filters, num_filters, num_heads, num_bond_types, bottleneck=bottleneck
            )
        self.edge_pred = GBLinear(num_filters[0], num_filters[1], num_bond_types + 1, 1, bottleneck=bottleneck)

        if with_root:
            self.root_lin = GBLinear(in_sca, in_vec, num_filters[0], num_filters[1], bottleneck=bottleneck)
            self.root_vector_expansion = EdgeExpansion(edge_channels)

        self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=edge_channels)
        self.distance_expansion_3A = GaussianSmearing(stop=3., num_gaussians=edge_channels)
        self.vector_expansion = EdgeExpansion(edge_channels)  # Linear(in_features=1, out_features=edge_channels, bias=False)

    def forward(self, pos_query, edge_index_query, cpx_pos, node_attr_compose, edge_index_q_cps_knn,
                index_real_cps_edge_for_atten=[], tri_edge_index=[], tri_edge_feat=[], atom_type_emb=None):

        vec_ij = pos_query[edge_index_q_cps_knn[0]] - cpx_pos[edge_index_q_cps_knn[1]]
        dist_ij = torch.norm(vec_ij, p=2, dim=-1).view(-1, 1)  # (A, 1)
        edge_ij = self.distance_expansion(dist_ij), self.vector_expansion(vec_ij)
        
        # node_attr_ctx_j = [node_attr_ctx_[edge_index_q_cps_knn[1]] for node_attr_ctx_ in node_attr_ctx]  # (A, H)
        h = self.message_module(node_attr_compose, edge_ij, edge_index_q_cps_knn[1], dist_ij, annealing=True)
        # Aggregate messages
        y = [scatter_add(h[0], index=edge_index_q_cps_knn[0], dim=0, dim_size=pos_query.size(0)), # (N_query, F)
             scatter_add(h[1], index=edge_index_q_cps_knn[0], dim=0, dim_size=pos_query.size(0))]
        # add information of new atom
        if isinstance(atom_type_emb, torch.Tensor):
            root_vec_ij = self.root_vector_expansion(pos_query)
            y_root_sca, y_root_vec = self.root_lin([atom_type_emb, root_vec_ij])
            y = [y_root_sca+y[0], y_root_vec+y[1]]

        if (len(edge_index_query) != 0) and (edge_index_query.size(1) > 0):
            # print(edge_index_query.shape)
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
            edge_attr = self.edge_feat(edge_attr)  # (E, N_edgetype)
            edge_attr = self.edge_atten(edge_attr, edge_index_query, cpx_pos, index_real_cps_edge_for_atten, tri_edge_index, tri_edge_feat)
            edge_pred, _ = self.edge_pred(edge_attr)
        else:
            edge_pred = torch.empty([0, self.num_bond_types+1], device=pos_query.device)
        
        return edge_pred


class ST_AttEdge_Exp(torch.nn.Module):
    def __init__(self, in_sca, in_vec, edge_channels, num_filters, num_bond_types=3, 
                 num_heads=4, bottleneck=1, use_conv1d=False):
        super(ST_AttEdge_Exp, self).__init__()
        self.num_bond_types = num_bond_types

        self.nn_edge_ij = Sequential(
            GBPerceptronVN(
                edge_channels, edge_channels, num_filters[0], num_filters[1],
                bottleneck=bottleneck, use_conv1d=use_conv1d
                ),
            GBLinear(
                num_filters[0], num_filters[1], num_filters[0], num_filters[1],
                bottleneck=bottleneck, use_conv1d=use_conv1d
                )
        )

        self.edge_feat = Sequential(
            GBPerceptronVN(
                num_filters[0] * 2 + in_sca, num_filters[1] * 2 + in_vec, num_filters[0],
                num_filters[1], bottleneck=bottleneck, use_conv1d=use_conv1d
                ),
            GBLinear(
                num_filters[0], num_filters[1], num_filters[0], num_filters[1],
                bottleneck=bottleneck, use_conv1d=use_conv1d
                )
        )
        self.edge_atten = AttentionEdges(num_filters, num_filters, num_heads, num_bond_types)
        self.edge_pred = GBLinear(
            num_filters[0], num_filters[1], (num_bond_types + 1) * 2, 1, bottleneck=bottleneck,
            use_conv1d=use_conv1d
            )

        self.distance_expansion_3A = GaussianSmearing(stop=3., num_gaussians=edge_channels)
        self.vector_expansion = EdgeExpansion(edge_channels)  # Linear(in_features=1, out_features=edge_channels, bias=False)

    def forward(self, h_atom, pos_query, edge_index_query, cpx_pos, node_attr_compose, index_real_cps_edge_for_atten=[], 
                tri_edge_index=[], tri_edge_feat=[]):

        if (len(edge_index_query) != 0) and (edge_index_query.size(1) > 0):
            # print(edge_index_query.shape)
            idx_node_i = edge_index_query[0]
            node_mol_i = [
                h_atom[0][idx_node_i],
                h_atom[1][idx_node_i]
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
            edge_attr = self.edge_feat(edge_attr)  # (E, N_edgetype)
            edge_attr = self.edge_atten(edge_attr, edge_index_query, cpx_pos, index_real_cps_edge_for_atten, tri_edge_index, tri_edge_feat)
            edge_pred, _ = self.edge_pred(edge_attr)
            s_edge, t_edge = edge_pred[:,:self.num_bond_types+1], edge_pred[:,self.num_bond_types+1:]
        else:
            s_edge = torch.empty([0, self.num_bond_types+1], device=pos_query.device)
            t_edge = torch.empty([0, self.num_bond_types+1], device=pos_query.device)
        return s_edge, t_edge
        


class BondFlow(torch.nn.Module):
    def __init__(self, in_sca, in_vec, edge_channels, num_filters, num_bond_types, 
                 num_heads=4, cutoff=10.0, with_root=True, num_st_layers=3, 
                 bottleneck=1, use_conv1d=False):
        super(BondFlow, self).__init__()
        self.with_root = with_root
        self.num_bond_types = num_bond_types
        self.num_st_layers = num_st_layers

        self.pos_encoder = PositionEncoder(
            in_sca, in_vec, edge_channels, num_filters, cutoff=cutoff, bottleneck=bottleneck,
            use_conv1d=use_conv1d
            )
        self.pos_filter = torch.nn.Sequential(
            GBPerceptronVN(
                num_filters[0], num_filters[1], num_filters[0], num_filters[1],
                bottleneck=bottleneck, use_conv1d=use_conv1d
                ),
            GBLinear(
                num_filters[0], num_filters[1], 1, 1, bottleneck=bottleneck, use_conv1d=use_conv1d
                )
            )
        if with_root:
            self.root_lin = GBLinear(
                in_sca, in_vec, num_filters[0], num_filters[1], bottleneck=bottleneck, use_conv1d=use_conv1d
                )
            self.root_vector_expansion = EdgeExpansion(edge_channels)

        self.flow_layers = torch.nn.ModuleList()
        for _ in range(num_st_layers):
            flow_layer = ST_AttEdge_Exp(
                in_sca, in_vec, edge_channels, 
                num_filters, num_bond_types=num_bond_types, num_heads=num_heads,
                bottleneck=bottleneck, use_conv1d=use_conv1d
                )
            self.flow_layers.append(flow_layer)

    def forward(self, z_edge, pos_query, edge_index_query, cpx_pos, node_attr_compose, edge_index_q_cps_knn,
                index_real_cps_edge_for_atten=[], tri_edge_index=[], tri_edge_feat=[], atom_type_emb=None,
                annealing=False):
        y = self.pos_encoder(
            pos_query, edge_index_q_cps_knn, cpx_pos, node_attr_compose, atom_type_emb,
            annealing=annealing
            )
        '''if isinstance(atom_type_emb, torch.Tensor) and self.with_root:
            root_vec_ij = self.root_vector_expansion(pos_query)
            y_root_sca, y_root_vec = self.root_lin([atom_type_emb, root_vec_ij])
            y = [y_root_sca+y[0], y_root_vec+y[1]]'''
        for ix in range(self.num_st_layers):
            s, t = self.flow_layers[ix](
                y, pos_query, edge_index_query, cpx_pos, node_attr_compose, 
                index_real_cps_edge_for_atten=index_real_cps_edge_for_atten, 
                tri_edge_index=tri_edge_index, tri_edge_feat=tri_edge_feat
                )
            s = s.exp()
            z_edge = (z_edge + t) * s
            if ix == 0:
                edge_log_jacob = (torch.abs(s) + 1e-20).log()
            else:
                edge_log_jacob += (torch.abs(s) + 1e-20).log()
        return z_edge, edge_log_jacob
    
    def reverse(self, edge_latent, pos_query, edge_index_query, cpx_pos, node_attr_compose, edge_index_q_cps_knn,
                index_real_cps_edge_for_atten=[], tri_edge_index=[], tri_edge_feat=[], atom_type_emb=None,
                annealing=False):
        y = self.pos_encoder(
            pos_query, edge_index_q_cps_knn, cpx_pos, node_attr_compose, atom_type_emb,
            annealing=annealing
            )
        '''if isinstance(atom_type_emb, torch.Tensor) and self.with_root:
            root_vec_ij = self.root_vector_expansion(pos_query)
            y_root_sca, y_root_vec = self.root_lin([atom_type_emb, root_vec_ij])
            y = [y_root_sca+y[0], y_root_vec+y[1]]'''
        for ix in range(self.num_st_layers):
            s, t = self.flow_layers[ix](
                y, pos_query, edge_index_query, cpx_pos, node_attr_compose, 
                index_real_cps_edge_for_atten=index_real_cps_edge_for_atten, 
                tri_edge_index=tri_edge_index, tri_edge_feat=tri_edge_feat
                )
            if s.size(0)==0 and t.size(0)==0:
                break
            else:
                edge_latent = (edge_latent / s.exp()) - t
        if s.size(0)==0 and t.size(0)==0:
            return torch.empty([0, self.num_bond_types+1], device=pos_query.device)
        else:
            return edge_latent
    
    def pos_classfier(self, pos_query, edge_index_q_cps_knn, cpx_pos, node_attr_compose, annealing=False):
        y = self.pos_encoder(
            pos_query, edge_index_q_cps_knn, cpx_pos, node_attr_compose, annealing=annealing
            )
        pred = self.pos_filter(y)
        return pred


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
        self.root_lin = GBLinear(
            in_sca, in_vec, num_filters[0], num_filters[1], bottleneck=bottleneck, use_conv1d=use_conv1d
            )
        self.root_vector_expansion = EdgeExpansion(edge_channels)
        
        #self.act_sca = LeakyReLU()
        #self.act_vec = VNLeakyReLU(hidden_channels[1], share_nonlinearity=True)    # 2023.1.13
        '''self.out_transform = GBLinear(
            num_filters[0], num_filters[1], num_filters[0], num_filters[1],
            bottleneck=bottleneck, use_conv1d=use_conv1d
        )
        self.layernorm_sca = nn.LayerNorm([num_filters[0]])
        self.layernorm_vec = nn.LayerNorm([num_filters[1], 3])'''   # 2023.1.13 注释掉
    
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
        '''# non-linear
        out_sca = self.layernorm_sca(y[0])
        out_vec = self.layernorm_vec(y[1])
        y = self.out_transform((self.act_sca(out_sca), self.act_vec(out_vec)))'''    # 2023.1.3
        return y


class BondFlowNew(Module):
    def __init__(self, in_sca, in_vec, edge_channels, num_filters, num_bond_types=3, num_heads=4,
                 cutoff=10.0, with_root=True, num_st_layers=6, bottleneck=1, use_conv1d=False):
        super(BondFlowNew, self).__init__()
        self.with_root = with_root
        self.num_bond_types = num_bond_types
        self.num_st_layers = num_st_layers
        ## query encoder
        self.pos_encoder = PositionEncoder(
            in_sca, in_vec, edge_channels, num_filters, cutoff=cutoff, bottleneck=bottleneck,
            use_conv1d=use_conv1d
            )
        ## query filter
        '''self.pos_filter = torch.nn.Sequential(
            GBPerceptronVN(
                num_filters[0], num_filters[1], num_filters[0], num_filters[1],
                bottleneck=bottleneck, use_conv1d=use_conv1d
                ),
            GBLinear(
                num_filters[0], num_filters[1], 1, 1, bottleneck=bottleneck, use_conv1d=use_conv1d
                )
            )'''
        ## edge pred
        self.distance_expansion_3A = GaussianSmearing(stop=3., num_gaussians=edge_channels)
        self.vector_expansion = EdgeExpansion(edge_channels)
        self.nn_edge_ij = Sequential(
            GBPerceptronVN(edge_channels, edge_channels, num_filters[0], num_filters[1]),
            GBLinear(num_filters[0], num_filters[1], num_filters[0], num_filters[1])
        )
        self.edge_feat = Sequential(
            GBPerceptronVN(num_filters[0] * 2 + in_sca, num_filters[1] * 2 + in_vec, num_filters[0], num_filters[1]),
            GBLinear(num_filters[0], num_filters[1], num_filters[0], num_filters[1])
        )
        self.edge_atten = AttentionEdges(num_filters, num_filters, num_heads, num_bond_types)
        ## flow layer
        self.flow_layers = torch.nn.ModuleList()
        for _ in range(num_st_layers):
            flow_layer = ST_GBP_Exp(
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
