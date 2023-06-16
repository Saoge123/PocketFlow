import torch
import torch.nn.functional as F
from torch.nn import Module, Linear, LeakyReLU, ModuleList, LayerNorm
import numpy as np
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from torch_scatter import scatter_sum, scatter_softmax
from math import pi as PI
from .net_utils import GaussianSmearing, EdgeExpansion, Rescale


EPS = 1e-6
class GDBLinear(Module):
    def __init__(self, in_scalar, in_vector, out_scalar, out_vector, bottleneck=(1,1), use_conv1d=False):
        super(GDBLinear, self).__init__()
        if isinstance(bottleneck, int):
            sca_bottleneck = bottleneck
            vec_bottleneck = bottleneck
        else:
            sca_bottleneck = bottleneck[0]
            vec_bottleneck = bottleneck[1]
        assert in_vector % vec_bottleneck == 0,\
            f"Input channel of vector ({in_vector}) must be divisible with bottleneck factor ({vec_bottleneck})"
        assert in_scalar % sca_bottleneck == 0,\
            f"Input channel of vector ({in_scalar}) must be divisible with bottleneck factor ({sca_bottleneck})"
        if sca_bottleneck > 1:
            self.sca_hidden_dim = in_scalar // sca_bottleneck
        else:
            self.sca_hidden_dim = max(in_vector, out_vector)

        if vec_bottleneck > 1:
            self.hidden_dim = in_vector // vec_bottleneck
        else:
            self.hidden_dim = max(in_vector, out_vector)

        self.out_vector = out_vector
        self.lin_vector = VNLinear(in_vector, self.hidden_dim, bias=False)
        self.lin_vector2 = VNLinear(self.hidden_dim, out_vector, bias=False)

        self.use_conv1d = use_conv1d
        self.scalar_to_vector_gates = Linear(out_scalar, out_vector)
        self.lin_scalar_1 = Linear(in_scalar, self.sca_hidden_dim, bias=False)
        self.lin_scalar_2 = Linear(self.hidden_dim + self.sca_hidden_dim, out_scalar, bias=False)

    def forward(self, features):
        feat_scalar, feat_vector = features
        feat_vector_inter = self.lin_vector(feat_vector)  # (N_samples, dim_hid, 3)
        feat_vector_norm = torch.norm(feat_vector_inter, p=2, dim=-1)  # (N_samples, dim_hid)
        z_sca = self.lin_scalar_1(feat_scalar)
        feat_scalar_cat = torch.cat([feat_vector_norm, z_sca], dim=-1)  # (N_samples, dim_hid+in_scalar)

        #z_sca = self.lin_scalar_1(feat_scalar_cat)
        out_scalar = self.lin_scalar_2(feat_scalar_cat)
        gating = torch.sigmoid(self.scalar_to_vector_gates(out_scalar)).unsqueeze(-1)
        
        out_vector = self.lin_vector2(feat_vector_inter)
        out_vector = gating * out_vector
        return out_scalar, out_vector


class GDBPerceptronVN(Module):
    def __init__(self, in_scalar, in_vector, out_scalar, out_vector, bottleneck=1, use_conv1d=False):
        super(GDBPerceptronVN, self).__init__()
        self.gb_linear = GDBLinear(
            in_scalar, in_vector, out_scalar, out_vector, bottleneck=bottleneck, use_conv1d=use_conv1d
            )
        self.act_sca = LeakyReLU()
        self.act_vec = VNLeakyReLU(out_vector)

    def forward(self, x):
        sca, vec = self.gb_linear(x)
        vec = self.act_vec(vec)
        sca = self.act_sca(sca)
        return sca, vec


class VNLinear(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super(VNLinear, self).__init__()
        self.map_to_feat = nn.Linear(in_channels, out_channels, *args, **kwargs)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_samples, N_feat, 3]
        '''
        x_out = self.map_to_feat(x.transpose(-2,-1)).transpose(-2,-1)
        return x_out


class VNLeakyReLU(nn.Module):
    def __init__(self, in_channels, share_nonlinearity=False, negative_slope=0.01):
        super(VNLeakyReLU, self).__init__()
        if share_nonlinearity == True:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)
        self.negative_slope = negative_slope

    def forward(self, x):
        '''
        x: point features of shape [B, N_samples, N_feat, 3]
        '''
        d = self.map_to_dir(x.transpose(-2,-1)).transpose(-2,-1)  # (N_samples, N_feat, 3)
        dotprod = (x*d).sum(-1, keepdim=True)  # sum over 3-value dimension
        mask = (dotprod >= 0).to(x.dtype)
        d_norm_sq = (d*d).sum(-1, keepdim=True)  # sum over 3-value dimension
        x_out = (self.negative_slope * x +
                (1-self.negative_slope) * (mask*x + (1-mask)*(x-(dotprod/(d_norm_sq+EPS))*d)))
        return x_out


class ST_GDBP_Exp(nn.Module):
    def __init__(self, in_scalar, in_vector, out_scalar, out_vector, bottleneck=1, use_conv1d=False):
        super(ST_GDBP_Exp, self).__init__()
        self.in_scalar = in_scalar
        self.in_vector = in_vector
        self.out_scalar = out_scalar
        self.out_vector = out_vector

        self.gb_linear1 = GDBLinear(
            in_scalar, in_vector, in_scalar, in_vector, bottleneck=bottleneck, use_conv1d=use_conv1d
            )
        self.gb_linear2 = GDBLinear(
            in_scalar, in_vector, out_scalar*2, out_vector, bottleneck=bottleneck, use_conv1d=use_conv1d
            )
        self.act_sca = nn.Tanh()
        self.act_vec = VNLeakyReLU(out_vector)
        self.rescale = Rescale()

    def forward(self, x):
        '''
        :param x: (batch * repeat_num for node/edge, emb)
        :return: w and b for affine operation
        '''
        sca, vec = self.gb_linear1(x)
        sca = self.act_sca(sca)
        vec = self.act_vec(vec)
        sca, vec = self.gb_linear2([sca, vec])
        s = sca[:, :self.out_scalar]
        t = sca[:, self.out_scalar:]
        s = self.rescale(torch.tanh(s))
        return s, t


class MessageAttention(Module):
    def __init__(self, in_sca, in_vec, out_sca, out_vec, bottleneck=1, num_heads=1, use_conv1d=False) -> None:
        super(MessageAttention, self).__init__()

        assert (in_sca % num_heads == 0) and (in_vec % num_heads == 0)
        assert (out_sca % num_heads == 0) and (out_vec % num_heads == 0)

        self.num_heads =num_heads
        self.lin_v = GDBLinear(in_sca, in_vec, out_sca, out_vec, bottleneck=bottleneck, use_conv1d=use_conv1d)
        self.lin_k = GDBLinear(in_sca, in_vec, out_sca, out_vec, bottleneck=bottleneck, use_conv1d=use_conv1d)

    def forward(self, x, query, edge_index_i):
        N = x[0].size(0)
        N_msg = len(edge_index_i)
        msg = [
            query[0].view(N_msg, self.num_heads, -1),
            query[1].view(N_msg, self.num_heads, -1, 3)
            ]
        k = self.lin_k(x)
        x_i = [
            k[0][edge_index_i].view(N_msg, self.num_heads, -1),
            k[1][edge_index_i].view(N_msg, self.num_heads, -1, 3)
            ]
        #alpha_scale = [x_i[0].size(-1)**0.5, x_i[1].size(-2)**0.5]
        alpha = [
            (msg[0] * x_i[0]).sum(-1), #/alpha_scale[0] # (N', heads)
            (msg[1] * x_i[1]).sum(-1).sum(-1) #/alpha_scale[1] # (N', heads)
            ]
        alpha = [
            scatter_softmax(alpha[0], edge_index_i, dim=0),
            scatter_softmax(alpha[1], edge_index_i, dim=0)
            ]
        msg = [
            (alpha[0].unsqueeze(-1) * msg[0]).view(N_msg, -1),
            (alpha[1].unsqueeze(-1).unsqueeze(-1) * msg[1]).view(N_msg, -1, 3)
            ]
        sca_msg = scatter_sum(msg[0], edge_index_i, dim=0, dim_size=N)
        vec_msg = scatter_sum(msg[1], edge_index_i, dim=0, dim_size=N)
        #return sca_msg, vec_msg
        root_sca, root_vec = self.lin_v(x)
        out_sca = sca_msg + root_sca
        out_vec = vec_msg + root_vec
        return out_sca, out_vec


class MessageModule(nn.Module):
    def __init__(self, node_sca, node_vec, edge_sca, edge_vec, out_sca, out_vec, 
                 bottleneck=1, cutoff=10., use_conv1d=False):
        super(MessageModule, self).__init__()
        hid_sca, hid_vec = edge_sca, edge_vec
        self.cutoff = cutoff
        self.node_gblinear = GDBLinear(
            node_sca, node_vec, out_sca, out_vec, bottleneck=bottleneck, use_conv1d=use_conv1d
            )
        self.edge_gbp = GDBPerceptronVN(
            edge_sca, edge_vec, hid_sca, hid_vec, bottleneck=bottleneck, use_conv1d=use_conv1d
            )

        self.sca_linear = Linear(hid_sca, out_sca)  # edge_sca for y_sca
        self.e2n_linear = Linear(hid_sca, out_vec)
        self.n2e_linear = Linear(out_sca, out_vec)
        self.edge_vnlinear = VNLinear(hid_vec, out_vec)

        self.out_gblienar = GDBLinear(
            out_sca, out_vec, out_sca, out_vec, bottleneck=bottleneck, use_conv1d=use_conv1d
            )

    def forward(self, node_features, edge_features, edge_index_node, dist_ij=None, annealing=False):
        node_scalar, node_vector = self.node_gblinear(node_features)
        node_scalar, node_vector = node_scalar[edge_index_node], node_vector[edge_index_node]
        edge_scalar, edge_vector = self.edge_gbp(edge_features)

        y_scalar = node_scalar * self.sca_linear(edge_scalar)
        y_node_vector = self.e2n_linear(edge_scalar).unsqueeze(-1) * node_vector
        y_edge_vector = self.n2e_linear(node_scalar).unsqueeze(-1) * self.edge_vnlinear(edge_vector)
        y_vector = y_node_vector + y_edge_vector

        output = self.out_gblienar((y_scalar, y_vector))

        if annealing:
            C = 0.5 * (torch.cos(dist_ij * PI / self.cutoff) + 1.0)  # (A, 1)
            C = C * (dist_ij <= self.cutoff) * (dist_ij >= 0.0)
            output = [output[0] * C.view(-1, 1), output[1] * C.view(-1, 1, 1)]   # (A, 1)
        return output


class AttentionInteractionBlockVN(Module):

    def __init__(self, hidden_channels, edge_channels, num_edge_types, bottleneck=1, num_heads=1,
                 cutoff=10., use_conv1d=False):
        super(AttentionInteractionBlockVN, self).__init__()
        self.num_heads = num_heads
        # edge features
        self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=edge_channels - num_edge_types)
        self.vector_expansion = EdgeExpansion(edge_channels)  # Linear(in_features=1, out_features=edge_channels, bias=False)
        ## compare encoder and classifier message passing

        # edge weigths and linear for values
        self.message_module = MessageModule(hidden_channels[0], hidden_channels[1], edge_channels, edge_channels,
                                            hidden_channels[0], hidden_channels[1], bottleneck=bottleneck,
                                            cutoff=cutoff, use_conv1d=use_conv1d)
        self.msg_att = MessageAttention(hidden_channels[0], hidden_channels[1], hidden_channels[0], hidden_channels[1],
                                        bottleneck=bottleneck, num_heads=num_heads, use_conv1d=use_conv1d)
        
        # centroid nodes and finall linear
        self.act_sca = LeakyReLU()
        self.act_vec = VNLeakyReLU(hidden_channels[1], share_nonlinearity=True)
        self.out_transform = GDBLinear(
            hidden_channels[0], hidden_channels[1], hidden_channels[0], hidden_channels[1], use_conv1d=use_conv1d,
            bottleneck=bottleneck
            )
        self.layernorm_sca = LayerNorm([hidden_channels[0]])
        self.layernorm_vec = LayerNorm([hidden_channels[1], 3])

    def forward(self, x, edge_index, edge_feature, edge_vector, edge_dist, annealing=False):
        """
        Args:
            x:  Node features: scalar features (N, feat), vector features(N, feat, 3)
            edge_index: (2, E).
            edge_attr:  (E, H)
        """
        scalar, vector = x
        N = scalar.size(0)
        row, col = edge_index   # (E,) , (E,)

        # Compute edge features
        #edge_dist = torch.norm(edge_vector, dim=-1, p=2)
        edge_sca_feat = torch.cat([self.distance_expansion(edge_dist), edge_feature], dim=-1)
        edge_vec_feat = self.vector_expansion(edge_vector) 

        msg_j_sca, msg_j_vec = self.message_module(
            x, (edge_sca_feat, edge_vec_feat), col, edge_dist, annealing=annealing
            )
        out_sca, out_vec = self.msg_att(x, (msg_j_sca, msg_j_vec), row)
        # non-linear
        out_sca = self.layernorm_sca(out_sca)
        out_vec = self.layernorm_vec(out_vec)
        out = self.out_transform((self.act_sca(out_sca), self.act_vec(out_vec)))
        return out


class AttentionEdges(Module):
    def __init__(self, hidden_channels, key_channels, num_heads=1, num_bond_types=3, bottleneck=1,
                 use_conv1d=False):
        super(AttentionEdges, self).__init__()
        
        assert (hidden_channels[0] % num_heads == 0) and (hidden_channels[1] % num_heads == 0)
        assert (key_channels[0] % num_heads == 0) and (key_channels[1] % num_heads == 0)

        self.hidden_channels = hidden_channels
        self.key_channels = key_channels
        self.num_heads = num_heads

        # linear transformation for attention 
        self.q_lin = GDBLinear(
            hidden_channels[0], hidden_channels[1], key_channels[0], key_channels[1],
            bottleneck=bottleneck, use_conv1d=use_conv1d
            )
        self.k_lin = GDBLinear(
            hidden_channels[0], hidden_channels[1], key_channels[0], key_channels[1],
            bottleneck=bottleneck, use_conv1d=use_conv1d
            )
        self.v_lin = GDBLinear(
            hidden_channels[0], hidden_channels[1], hidden_channels[0], hidden_channels[1],
            bottleneck=bottleneck, use_conv1d=use_conv1d
            )

        self.atten_bias_lin = AttentionBias(
            self.num_heads, hidden_channels, num_bond_types=num_bond_types,
            bottleneck=bottleneck, use_conv1d=use_conv1d
            )
        
        self.layernorm_sca = LayerNorm([hidden_channels[0]])
        self.layernorm_vec = LayerNorm([hidden_channels[1], 3])

    def forward(self, edge_attr, edge_index, pos_compose, 
                          index_real_cps_edge_for_atten, tri_edge_index, tri_edge_feat,):
        """
        Args:
            x:  edge features: scalar features (N, feat), vector features(N, feat, 3)
            edge_attr:  (E, H)
            edge_index: (2, E). the row can be seen as batch_edge
        """
        scalar, vector = edge_attr
        N = scalar.size(0)
        row, col = edge_index   # (N,) 

        # Project to multiple key, query and value spaces
        h_queries = self.q_lin(edge_attr)
        h_queries = (h_queries[0].view(N, self.num_heads, -1),  # (N, heads, K_per_head)
                    h_queries[1].view(N, self.num_heads, -1, 3))  # (N, heads, K_per_head, 3)
        h_keys = self.k_lin(edge_attr)
        h_keys = (h_keys[0].view(N, self.num_heads, -1),  # (N, heads, K_per_head)
                    h_keys[1].view(N, self.num_heads, -1, 3))  # (N, heads, K_per_head, 3)
        h_values = self.v_lin(edge_attr)
        h_values = (h_values[0].view(N, self.num_heads, -1),  # (N, heads, K_per_head)
                    h_values[1].view(N, self.num_heads, -1, 3))  # (N, heads, K_per_head, 3)

        # assert (index_edge_i_list == index_real_cps_edge_for_atten[0]).all()
        # assert (index_edge_j_list == index_real_cps_edge_for_atten[1]).all()
        index_edge_i_list, index_edge_j_list = index_real_cps_edge_for_atten

        # # get nodes of triangle edges

        atten_bias = self.atten_bias_lin(
            tri_edge_index,
            tri_edge_feat,
            pos_compose,
        )


        # query * key
        queries_i = [h_queries[0][index_edge_i_list], h_queries[1][index_edge_i_list]]
        keys_j = [h_keys[0][index_edge_j_list], h_keys[1][index_edge_j_list]]

        qk_ij = [
            (queries_i[0] * keys_j[0]).sum(-1),  # (N', heads)
            (queries_i[1] * keys_j[1]).sum(-1).sum(-1)  # (N', heads)
            ]

        alpha = [
            atten_bias[0] + qk_ij[0],
            atten_bias[1] + qk_ij[1]
            ]
        alpha = [
            scatter_softmax(alpha[0], index_edge_i_list, dim=0),  # (N', heads)
            scatter_softmax(alpha[1], index_edge_i_list, dim=0)  # (N', heads)
            ] 

        values_j = [h_values[0][index_edge_j_list], h_values[1][index_edge_j_list]]
        num_attens = len(index_edge_j_list)
        output =[
            scatter_sum((alpha[0].unsqueeze(-1) * values_j[0]).view(num_attens, -1), index_edge_i_list, dim=0, dim_size=N),   # (N, H, 3)
            scatter_sum((alpha[1].unsqueeze(-1).unsqueeze(-1) * values_j[1]).view(num_attens, -1, 3), index_edge_i_list, dim=0, dim_size=N)   # (N, H, 3)
        ]

        # output 
        output = [edge_attr[0] + output[0], edge_attr[1] + output[1]]
        output = [self.layernorm_sca(output[0]), self.layernorm_vec(output[1])]

        return output


class AttentionBias(Module):
    def __init__(self, num_heads, hidden_channels, cutoff=10., num_bond_types=3,
                 bottleneck=1, use_conv1d=False): #TODO: change the cutoff
        super(AttentionBias, self).__init__()
        num_edge_types = num_bond_types + 1
        self.num_bond_types = num_bond_types
        self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=hidden_channels[0] - num_edge_types-1)  # minus 1 for self edges (e.g. edge 0-0)
        self.vector_expansion = EdgeExpansion(hidden_channels[1])  # Linear(in_features=1, out_features=hidden_channels[1], bias=False)
        self.gblinear = GDBLinear(
            hidden_channels[0], hidden_channels[1], num_heads, num_heads,
            bottleneck=bottleneck, use_conv1d=use_conv1d
            )

    def forward(self,  tri_edge_index, tri_edge_feat, pos_compose):
        node_a, node_b = tri_edge_index
        pos_a = pos_compose[node_a]
        pos_b = pos_compose[node_b]
        vector = pos_a - pos_b
        dist = torch.norm(vector, p=2, dim=-1)
        
        dist_feat = self.distance_expansion(dist)
        sca_feat = torch.cat([
            dist_feat,
            tri_edge_feat,
        ], dim=-1)
        vec_feat = self.vector_expansion(vector)
        output_sca, output_vec = self.gblinear([sca_feat, vec_feat])
        output_vec = (output_vec * output_vec).sum(-1)
        return output_sca, output_vec