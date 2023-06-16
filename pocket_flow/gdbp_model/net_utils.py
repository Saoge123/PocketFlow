import torch
from torch import nn
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F



keys = ['edge_flow.flow_layers.5', 'atom_flow.flow_layers.5', 
        'pos_predictor.mu_net', 'pos_predictor.logsigma_net', 'pos_predictor.pi_net',
        'focal_net.net']
def reset_parameters(model, keys):
    for name, para in model.named_parameters():
        for k in keys:
            if k in name and 'bias' in name:
                torch.nn.init.constant_(para, 0.)
            elif k in name and 'layernorm' in name:
                torch.nn.init.constant_(para, 1.)
            elif k in name and 'rescale.weight' in name:
                torch.nn.init.constant_(para, 0.)
            elif k in name:
                torch.nn.init.kaiming_normal_(para)
    return model

def freeze_parameters(model, keys):
    for name, para in model.named_parameters():
        for k in keys:
            if k in name:
                para.requires_grad = False
    return model


def flow_reverse(flow_layers, latent, feat):
    for i in reversed(range(len(flow_layers))):
        s_sca, t_sca, vec = flow_layers[i](feat)
        s_sca = s_sca.exp()
        latent = (latent / s_sca) - t_sca
    return latent, vec


def flow_forward(flow_layers, x_z, feature):
    for i in range(len(flow_layers)):
        s_sca, t_sca, vec = flow_layers[i](feature)
        s_sca = s_sca.exp()
        x_z = (x_z + t_sca) * s_sca
        if i == 0:
            x_log_jacob = (torch.abs(s_sca) + 1e-20).log()
        else:
            x_log_jacob += (torch.abs(s_sca) + 1e-20).log()
    return x_z, x_log_jacob, vec


class GaussianSmearing(nn.Module):
    def __init__(self, start=0.0, stop=10.0, num_gaussians=50):
        super(GaussianSmearing, self).__init__()
        self.stop = stop
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.clamp_max(self.stop)
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class EdgeExpansion(nn.Module):
    def __init__(self, edge_channels):
        super(EdgeExpansion, self).__init__()
        self.nn = nn.Linear(in_features=1, out_features=edge_channels, bias=False)
    
    def forward(self, edge_vector):
        edge_vector = edge_vector / (torch.norm(edge_vector, p=2, dim=1, keepdim=True)+1e-7)
        expansion = self.nn(edge_vector.unsqueeze(-1)).transpose(1, -1)
        return expansion


class Scalarize(nn.Module):
    def __init__(self, sca_in_dim, vec_in_dim, hidden_dim, out_dim, act_fn=nn.Sigmoid()) -> None:
        super(Scalarize, self).__init__()
        self.sca_in_dim = sca_in_dim
        self.vec_in_dim = vec_in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.lin_scalarize_1 = nn.Linear(sca_in_dim+vec_in_dim, hidden_dim)
        self.lin_scalarize_2 = nn.Linear(hidden_dim, out_dim)
        self.act_fn = act_fn
    
    def forward(self, x):
        sca, vec = x[0].view(-1, self.sca_in_dim), x[1]
        norm_vec = torch.norm(vec, p=2, dim=-1).view(-1, self.vec_in_dim)
        sca = torch.cat([sca, norm_vec], dim=1)
        sca = self.lin_scalarize_1(sca)
        sca = self.act_fn(sca)
        sca = self.lin_scalarize_2(sca)
        return sca


class Rescale(nn.Module):
    def __init__(self):
        super(Rescale, self).__init__()
        self.weight = nn.Parameter(torch.zeros([1]))

    def forward(self, x):
        if torch.isnan(torch.exp(self.weight)).any():
            print(self.weight)
            raise RuntimeError('Rescale factor has NaN entries')

        x = torch.exp(self.weight) * x
        return x



class AtomEmbedding(nn.Module):
    def __init__(self, in_scalar, in_vector,
                 out_scalar, out_vector, vector_normalizer=20.):
        super(AtomEmbedding, self).__init__()
        assert in_vector == 1
        self.in_scalar = in_scalar
        self.vector_normalizer = vector_normalizer
        self.emb_sca = nn.Linear(in_scalar, out_scalar)
        self.emb_vec = nn.Linear(in_vector, out_vector)

    def forward(self, scalar_input, vector_input):
        if isinstance(self.vector_normalizer, float):
            vector_input = vector_input / self.vector_normalizer
        else:
            vector_input = vector_input / torch.norm(vector_input, p=2, dim=-1)
        assert vector_input.shape[1:] == (3, ), 'Not support. Only one vector can be input'
        sca_emb = self.emb_sca(scalar_input[:, :self.in_scalar])  # b, f -> b, f'
        vec_emb = vector_input.unsqueeze(-1)  # b, 3 -> b, 3, 1
        vec_emb = self.emb_vec(vec_emb).transpose(1, -1)  # b, 1, 3 -> b, f', 3
        return sca_emb, vec_emb


def embed_compose(compose_feature, compose_pos, idx_ligand, idx_protein,
                  ligand_atom_emb, protein_atom_emb, emb_dim):
    h_ligand = ligand_atom_emb(compose_feature[idx_ligand], compose_pos[idx_ligand])
    h_protein = protein_atom_emb(compose_feature[idx_protein], compose_pos[idx_protein])
    
    h_sca = torch.zeros([len(compose_pos), emb_dim[0]],).to(h_ligand[0])
    h_vec = torch.zeros([len(compose_pos), emb_dim[1], 3],).to(h_ligand[1])
    h_sca[idx_ligand], h_sca[idx_protein] = h_ligand[0], h_protein[0]
    h_vec[idx_ligand], h_vec[idx_protein] = h_ligand[1], h_protein[1]
    return [h_sca, h_vec]


class SmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth_one_hot(targets:torch.Tensor, n_classes:int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                    device=targets.device) \
                .fill_(smoothing /(n_classes-1)) \
                .scatter_(1, targets.data.unsqueeze(1), 1.-smoothing)
        return targets

    def forward(self, inputs, targets):
        targets = SmoothCrossEntropyLoss._smooth_one_hot(targets, inputs.size(-1),
            self.smoothing)
        lsm = F.log_softmax(inputs, -1)

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()
        return loss
