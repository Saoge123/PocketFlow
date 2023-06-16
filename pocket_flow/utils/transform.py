import copy
from multiprocessing import context
# from multiprocessing import context
import os
import sys
sys.path.append('.')
import random
import time
import uuid
from itertools import compress
# from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn.pool import knn_graph
from torch_geometric.transforms import Compose
from torch_geometric.utils.subgraph import subgraph
from torch_geometric.nn import knn, radius
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add
# import multiprocessing as multi
# from torch_geometric.data import DataLoader

from .data import ComplexData

from typing import List, Callable, Union
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform
from .transform_utils import count_neighbors, mask_node, get_bfs_perm, get_rfs_perm, make_pos_label,\
                            get_complex_graph, get_complex_graph_, sample_edge_with_radius, get_tri_edges

class TrajCompose(BaseTransform):
    """Composes several transforms together.

    Args:
        transforms (List[Callable]): List of transforms to compose.
    """
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, data: Union[Data, HeteroData]):
        if isinstance(data, (list, tuple)):
            l = []
            for i in data:
                for transform in self.transforms:
                    traj = transform(i)
                l += traj
        else:
            for transform in self.transforms:
                data = transform(data)
        return data

    def __repr__(self) -> str:
        args = [f'  {transform}' for transform in self.transforms]
        return '{}([\n{}\n])'.format(self.__class__.__name__, ',\n'.join(args))


class RefineData(object):
    def __init__(self):
        super().__init__()

    def __call__(self, data):
        # delete H atom of pocket
        protein_element = data.protein_element
        is_H_protein = (protein_element == 1)
        if torch.sum(is_H_protein) > 0:
            not_H_protein = ~is_H_protein
            data.protein_atom_name = list(compress(data.protein_atom_name, not_H_protein)) 
            data.protein_atom_to_aa_type = data.protein_atom_to_aa_type[not_H_protein]
            data.protein_element = data.protein_element[not_H_protein]
            data.protein_is_backbone = data.protein_is_backbone[not_H_protein]
            data.protein_pos = data.protein_pos[not_H_protein]
        # delete H atom of ligand
        ligand_element = data.ligand_element
        is_H_ligand = (ligand_element == 1)
        if torch.sum(is_H_ligand) > 0:
            not_H_ligand = ~is_H_ligand
            data.ligand_atom_feature = data.ligand_atom_feature[not_H_ligand]
            data.ligand_element = data.ligand_element[not_H_ligand]
            data.ligand_pos = data.ligand_pos[not_H_ligand]
            # nbh
            index_atom_H = torch.nonzero(is_H_ligand)[:, 0]
            index_changer = -np.ones(len(not_H_ligand), dtype=np.int64)
            index_changer[not_H_ligand] = np.arange(torch.sum(not_H_ligand))
            new_nbh_list = [value for ind_this, value in zip(not_H_ligand, data.ligand_nbh_list.values()) if ind_this]
            data.ligand_nbh_list = {i:[index_changer[node] for node in neigh if node not in index_atom_H] for i, neigh in enumerate(new_nbh_list)}
            # bond
            ind_bond_with_H = np.array([(bond_i in index_atom_H) | (bond_j in index_atom_H) for bond_i, bond_j in zip(*data.ligand_bond_index)])
            ind_bond_without_H = ~ind_bond_with_H
            old_ligand_bond_index = data.ligand_bond_index[:, ind_bond_without_H]
            data.ligand_bond_index = torch.tensor(index_changer)[old_ligand_bond_index]
            data.ligand_bond_type = data.ligand_bond_type[ind_bond_without_H]
        return data

class LigandCountNeighbors(object):
    def __init__(self):
        super().__init__()

    def __call__(self, data):
        data.ligand_num_neighbors = count_neighbors(
            data.ligand_bond_index,
            symmetry=True,
            num_nodes=data.ligand_element.size(0),
        )
        data.ligand_atom_valence = count_neighbors(
            data.ligand_bond_index,
            symmetry=True,
            valence=data.ligand_bond_type,
            num_nodes=data.ligand_element.size(0),
        )
        data.ligand_atom_num_bonds = torch.stack([
            count_neighbors(
                data.ligand_bond_index,
                symmetry=True,
                valence=(data.ligand_bond_type == i).long(),
                num_nodes=data.ligand_element.size(0),
            ) for i in [1, 2, 3]
        ], dim = -1)
        return data


class FeaturizeProteinAtom(object):
    def __init__(self):
        super().__init__()
        # self.atomic_numbers = torch.LongTensor([1, 6, 7, 8, 16, 34])    # H, C, N, O, S, Se
        self.atomic_numbers = torch.LongTensor([6, 7, 8, 16, 34])    # H, C, N, O, S, Se
        self.max_num_aa = 20
    @property
    def feature_dim(self):
        return self.atomic_numbers.size(0) + self.max_num_aa + 1 + 1

    def __call__(self, data:ComplexData):
        element = data.protein_element.view(-1, 1) == self.atomic_numbers.view(1, -1)   # (N_atoms, N_elements)
        amino_acid = F.one_hot(data.protein_atom_to_aa_type, num_classes=self.max_num_aa)
        is_backbone = data.protein_is_backbone.view(-1, 1).long()
        is_mol_atom = torch.zeros_like(is_backbone, dtype=torch.long)
        # x = torch.cat([element, amino_acid, is_backbone], dim=-1)
        x = torch.cat([element, amino_acid, is_backbone, is_mol_atom], dim=-1)
        data.protein_atom_feature = x
        # data.compose_index = torch.arange(len(element), dtype=torch.long)
        return data


class FeaturizeLigandAtom(object):
    def __init__(self, atomic_numbers=[1,6,7,8,9,15,16,17,35,53]):
        super().__init__()
        # self.atomic_numbers = torch.LongTensor([1,6,7,8,9,15,16,17])  # H C N O F P S Cl
        self.atomic_numbers = torch.LongTensor(atomic_numbers)  # C N O F P S Cl
        #assert len(self.atomic_numbers) == 7, NotImplementedError('fix the staticmethod: chagne_bond')
        # @property
        # def num_properties(self):
            # return len(ATOM_FAMILIES)
    @property
    def feature_dim(self):
        return self.atomic_numbers.size(0) + (1 + 1 + 1) + 3

    def __call__(self, data:ComplexData):
        
        element = data.ligand_element.view(-1, 1) == self.atomic_numbers.view(1, -1)   # (N_atoms, N_elements)
        # chem_feature = data.ligand_atom_feature
        is_mol_atom = torch.ones([len(element), 1], dtype=torch.long)
        n_neigh = data.ligand_num_neighbors.view(-1, 1)
        n_valence = data.ligand_atom_valence.view(-1, 1)
        ligand_atom_num_bonds = data.ligand_atom_num_bonds
        # x = torch.cat([element, chem_feature, ], dim=-1)
        x = torch.cat([element, is_mol_atom, n_neigh, n_valence, ligand_atom_num_bonds], dim=-1)
        data.ligand_atom_feature_full = x
        return data


class LigandTrajectory(object):
    def __init__(self, perm_type='bfs', p=None, num_atom_type=10, y_pos_std=0.05):
        super().__init__()
        if perm_type not in {'rfs','bfs','mix'}:
            raise ValueError("perm_type should be the one of {'rfs','bfs','mix'}")
        self.perm_type = perm_type
        self.num_atom_type = num_atom_type
        self.y_pos_std = y_pos_std
        self.p = p

    def __call__(self, data):
        if self.perm_type == 'rfs':
            perm, edge_index = get_rfs_perm(data.ligand_nbh_list, data.ligand_ring_info)
        elif self.perm_type == 'bfs':
            perm, edge_index = get_bfs_perm(data.ligand_nbh_list)
        elif self.perm_type == 'mix':
            perm_type = np.random.choice(['rfs', 'bfs'], p=self.p)
            if perm_type == 'rfs':
                perm, edge_index = get_rfs_perm(data.ligand_nbh_list, data.ligand_ring_info)
            else:
                perm, edge_index = get_bfs_perm(data.ligand_nbh_list)
        traj =[]
        for ix, i in enumerate(perm):
            data_step = copy.deepcopy(data)
            if ix == 0:
                out = mask_node(data_step, torch.empty([0], dtype=torch.long), perm, 
                                num_atom_type=self.num_atom_type, y_pos_std=self.y_pos_std)
                #out.edge_index_step = torch.LongTensor(edge_index[ix]).permute(-1)
                traj.append(out)
            else:
                out = mask_node(data_step, perm[:ix], perm[ix:], num_atom_type=self.num_atom_type,
                                y_pos_std=self.y_pos_std)
                #out.edge_index_step = torch.LongTensor(edge_index[ix]).permute(-1,0)
                traj.append(out)
        del data
        return traj


class FocalMaker(object):
    def __init__(self, r=4.0, num_work=16, atomic_numbers=[1,6,7,8,9,15,16,17,35,53]) -> None:
        self.r = r
        self.num_work = num_work
        self.atomic_numbers = torch.LongTensor(atomic_numbers)

    def run(self, data):
        if data.ligand_context_pos.size(0) == 0:
            masked_pos = data.ligand_pos[data.masked_idx[0]]
            focal_idx_in_context = torch.norm(data.protein_pos-masked_pos.unsqueeze(0), p=2, dim=-1).argmin()
            data.focal_idx_in_context = focal_idx_in_context.unsqueeze(0)
            data.focal_idx_in_context_ = focal_idx_in_context.unsqueeze(0)
            data.atom_label = torch.nonzero(data.ligand_element[data.masked_idx[0]] == self.atomic_numbers).squeeze(0)
            data.edge_label = torch.empty(0, dtype=torch.long)
            data.focal_label = torch.empty(0, dtype=torch.long)
            data.edge_query_index_0 = torch.zeros_like(data.edge_label)
            data.edge_query_index_1 = torch.arange(data.edge_label.size(0))
            edge_index_query = torch.stack([data.edge_query_index_0, data.edge_query_index_1])
            data.index_real_cps_edge_for_atten, data.tri_edge_index, data.tri_edge_feat = get_tri_edges(
                                                                        edge_index_query, 
                                                                        data.y_pos, data.context_idx, 
                                                                        data.ligand_context_bond_index, 
                                                                        data.ligand_context_bond_type
                                                                        )
            # candidate focal atom without ligand
            assign_index = radius(x=data.ligand_masked_pos, y=data.protein_pos, r=self.r, num_workers=self.num_work)
            if assign_index.size(1) == 0:
                dist = torch.norm(data.protein_pos.unsqueeze(1) - data.ligand_masked_pos.unsqueeze(0), p=2, dim=-1)
                assign_index = torch.nonzero(dist <= torch.min(dist)+1e-5)[0:1].transpose(0, 1)
            data.candidate_focal_idx_in_protein = torch.unique(assign_index[0])
            candidate_focal_label_in_protein = torch.zeros(data.protein_pos.size(0), dtype=torch.bool)
            candidate_focal_label_in_protein[data.candidate_focal_idx_in_protein] = True
            data.candidate_focal_label_in_protein = candidate_focal_label_in_protein
            data.apo_protein_idx = torch.arange(data.protein_pos.size(0), dtype=torch.long)
        else:
            new_step_atom_idx = data.masked_idx[0]
            candidate_focal_idx_in_context = torch.LongTensor(data.ligand_nbh_list[new_step_atom_idx.item()])
            focal_idx_in_context_mask = (data.context_idx.unsqueeze(1) == candidate_focal_idx_in_context).any(1)
            data.focal_idx_in_context_ = torch.nonzero(focal_idx_in_context_mask).view(-1)
            #data.idx_focal_in_protein = torch.empty(0, dtype=torch.long)
            focal_choice_idx = torch.multinomial(torch.ones_like(data.focal_idx_in_context_).float(), 1)
            data.focal_idx_in_context = data.focal_idx_in_context_[focal_choice_idx]
            data.focal_label = torch.zeros_like(data.context_idx)
            data.focal_label[data.focal_idx_in_context] = 1
            data.atom_label = torch.nonzero(data.ligand_element[data.masked_idx[0]] == self.atomic_numbers).squeeze(0)
            data.apo_protein_idx = torch.empty(0, dtype=torch.long)
            data.candidate_focal_idx_in_protein = torch.empty(0, dtype=torch.long)
            data.candidate_focal_label_in_protein = torch.empty(0, dtype=torch.long)
            #focal_idx_in_ligand = data.context_idx[focal_idx_in_context_mask]
            # get edge label
            data = sample_edge_with_radius(data, r=4.0)
            # get triangle edge
            edge_index_query = torch.stack([data.edge_query_index_0, data.edge_query_index_1])
            data.index_real_cps_edge_for_atten, data.tri_edge_index, data.tri_edge_feat = get_tri_edges(
                                                                        edge_index_query, 
                                                                        data.y_pos, data.context_idx, 
                                                                        data.ligand_context_bond_index, 
                                                                        data.ligand_context_bond_type
                                                                        )
        return data

    def __call__(self, data_list):
        data_list_new = []
        for i in data_list:
            data_list_new.append(self.run(i))
        del data_list
        return data_list_new

        
class AtomComposer(object):
    def __init__(self, knn=16, num_workers=8, graph_type='knn', radius=10.0,
                 num_real_pos=5, num_fake_pos=5, pos_real_std=0.05, pos_fake_std=2.0,
                 for_gen=False, use_protein_bond=True): 
        assert graph_type in ['rad', 'knn'], "graph_type should be the one of ['rad', 'knn']"
        self.graph_type = graph_type
        self.radius = radius
        self.knn = knn
        self.num_workers = num_workers
        self.num_real_pos = num_real_pos
        self.num_fake_pos = num_fake_pos
        self.pos_real_std = pos_real_std
        self.pos_fake_std = pos_fake_std
        self.for_gen = for_gen
        self.use_protein_bond = use_protein_bond

    def run(self, data):
        protein_feat_dim = data.protein_atom_feature.size(-1)
        ligand_feat_dim = data.ligand_context_feature_full.size(-1)
        num_ligand_ctx_atom = data.ligand_context_pos.size(0)
        num_protein_atom = data.protein_pos.size(0)
        
        data.cpx_pos = torch.cat([data.ligand_context_pos, data.protein_pos], dim=0)
        data.step_batch = torch.zeros(data.cpx_pos.size(0), dtype=torch.long)
        num_complex_atom = data.cpx_pos.size(0)
        ligand_context_feature_full_expand = torch.cat([
            data.ligand_context_feature_full, 
            torch.zeros([num_ligand_ctx_atom, abs(protein_feat_dim-ligand_feat_dim)], dtype=torch.long)
        ], dim=1)
        data.cpx_feature = torch.cat([ligand_context_feature_full_expand, data.protein_atom_feature], dim=0)
        data.idx_ligand_ctx_in_cpx = torch.arange(num_ligand_ctx_atom, dtype=torch.long)  # can be delete
        data.idx_protein_in_cpx = torch.arange(num_protein_atom, dtype=torch.long) + num_ligand_ctx_atom  # can be delete
        if self.use_protein_bond:
            data = get_complex_graph_(data, knn=self.knn, num_workers=self.num_workers, graph_type=self.graph_type,
                                      radius=self.radius)
        else:
            data = get_complex_graph(data, num_ligand_ctx_atom, num_complex_atom, num_workers=self.num_workers,
                                 graph_type=self.graph_type, knn=self.knn, radius=self.radius)
        pos_query_knn_edge_idx = knn(x=data.cpx_pos, y=data.y_pos, k=self.knn, num_workers=16)
        data.pos_query_knn_edge_idx_0, data.pos_query_knn_edge_idx_1 = pos_query_knn_edge_idx
        data.cpx_backbone_index = torch.nonzero(data.protein_is_backbone).view(-1) + data.ligand_context_feature_full.size(0)
        if self.for_gen is False:
            data = make_pos_label(
                data, num_real_pos=self.num_real_pos, num_fake_pos=self.num_fake_pos, 
                pos_real_std=self.pos_real_std, pos_fake_std=self.pos_fake_std, k=self.knn
                )
        return data
    
    def __call__(self, data_list):
        d_list_new = []
        for d in data_list:
            d_list_new.append(self.run(d))
        del data_list
        return d_list_new


class Combine(object):
    def __init__(self, lig_traj, focal_maker, atom_composer, lig_only=False):
        self.lig_traj =lig_traj
        self.focal_maker = focal_maker
        self.atom_composer = atom_composer
        self.lig_only = lig_only
    
    def __call__(self, data):
        if self.lig_traj.perm_type == 'rfs':
            perm, edge_index = get_rfs_perm(data.ligand_nbh_list, data.ligand_ring_info)
        elif self.lig_traj.perm_type == 'bfs':
            perm, edge_index = get_bfs_perm(data.ligand_nbh_list)
        elif self.lig_traj.perm_type == 'mix':
            perm_type = np.random.choice(['rfs', 'bfs'], p=self.lig_traj.p)
            if perm_type == 'rfs':
                perm, edge_index = get_rfs_perm(data.ligand_nbh_list, data.ligand_ring_info)
            else:
                perm, edge_index = get_bfs_perm(data.ligand_nbh_list)
        traj =[]
        for ix, i in enumerate(perm):
            data_step = copy.deepcopy(data)
            if ix == 0:
                if self.lig_only == False:
                    out = mask_node(data_step, torch.empty([0], dtype=torch.long), perm, 
                                    num_atom_type=self.lig_traj.num_atom_type, 
                                    y_pos_std=self.lig_traj.y_pos_std)
                    #out.edge_index_step = torch.LongTensor(edge_index[ix]).permute(-1)
                else:
                    continue
            else:
                out = mask_node(data_step, perm[:ix], perm[ix:], 
                                num_atom_type=self.lig_traj.num_atom_type,
                                y_pos_std=self.lig_traj.y_pos_std)
                #out.edge_index_step = torch.LongTensor(edge_index[ix]).permute(-1,0)
            out = self.focal_maker.run(out)
            out = self.atom_composer.run(out)
            traj.append(out)
        del data
        return traj


COLLATE_KEYS = [#'context_idx', 'masked_idx', 'ligand_masked_element', 'ligand_masked_pos', 'ligand_context_element',
                #'ligand_context_feature_full', 'ligand_context_pos', 'ligand_context_bond_index', 'ligand_context_bond_type',
                #'ligand_context_num_neighbors', 'ligand_context_valence', 'ligand_context_num_bonds', 
                'cpx_pos', 'cpx_feature', 'idx_ligand_ctx_in_cpx', 'idx_protein_in_cpx', 'focal_idx_in_context', 
                'focal_idx_in_context_', 'cpx_edge_index', 'cpx_edge_type', 'cpx_edge_feature',  'pos_query_knn_edge_idx_0',
                'pos_query_knn_edge_idx_1', 'focal_label', 'ligand_frontier', 'y_pos', 'edge_label', 'atom_label', 'edge_query_index_0', 
                'edge_query_index_1', 'pos_fake', 'pos_fake_knn_edge_idx_0', 'pos_fake_knn_edge_idx_1', 'pos_real', 
                'pos_real_knn_edge_idx_0', 'pos_real_knn_edge_idx_1', 'index_real_cps_edge_for_atten', 'tri_edge_index', 
                'tri_edge_feat', 'cpx_backbone_index', 'cpx_pos_batch', 'y_pos_batch', 'edge_label_batch', 'atom_label_batch',
                'apo_protein_idx', 'candidate_focal_idx_in_protein', 'candidate_focal_label_in_protein', 'step_batch'
                ]# 'ligand_ctx_step']

def collate_fn(data_list, collate_keys=COLLATE_KEYS):
    if collate_keys:
        data_dict = {k:[] for k in collate_keys}
    else:
        data_dict = {k:[] for k in data_list[0].keys}
    # 
    data_dict['protein_pos'] = data_list[0].protein_pos
    data_dict['protein_atom_feature'] = data_list[0].protein_atom_feature
    data_dict['ligand_pos'] = data_list[0].ligand_pos
    data_dict['ligand_element'] = data_list[0].ligand_element
    data_dict['ligand_bond_index'] = data_list[0].ligand_bond_index
    data_dict['ligand_bond_type'] = data_list[0].ligand_bond_type
    data_dict['ligand_atom_feature_full'] = data_list[0].ligand_atom_feature_full
    
    compose_pos_cusum = 0
    edge_query_index_cusum = 0
    pos_fake_cusum = 0
    pos_real_cusum = 0
    #index_real_cps_edge_for_atten_cusum = 0
    for idx, d in enumerate(data_list):
        data_dict['cpx_pos'].append(d['cpx_pos'])
        data_dict['cpx_feature'].append(d['cpx_feature'])
        data_dict['idx_ligand_ctx_in_cpx'].append(d.idx_ligand_ctx_in_cpx+compose_pos_cusum)
        data_dict['idx_protein_in_cpx'].append(d.idx_protein_in_cpx+compose_pos_cusum)
        data_dict['candidate_focal_idx_in_protein'].append(d.candidate_focal_idx_in_protein)
        data_dict['candidate_focal_label_in_protein'].append(d.candidate_focal_label_in_protein)
        data_dict['apo_protein_idx'].append(d.apo_protein_idx)
        data_dict['focal_idx_in_context'].append(d.focal_idx_in_context+compose_pos_cusum)
        data_dict['focal_idx_in_context_'].append(d.focal_idx_in_context_+compose_pos_cusum)
        data_dict['cpx_edge_index'].append(d.cpx_edge_index+compose_pos_cusum)
        data_dict['cpx_edge_type'].append(d.cpx_edge_type)
        data_dict['cpx_edge_feature'].append(d.cpx_edge_feature)
        data_dict['cpx_backbone_index'].append(d.cpx_backbone_index+compose_pos_cusum)
        data_dict['focal_label'].append(d.focal_label)
        data_dict['y_pos'].append(d.y_pos)
        data_dict['ligand_frontier'].append(d.ligand_frontier)
        data_dict['edge_label'].append(d.edge_label)
        data_dict['atom_label'].append(d.atom_label)
        data_dict['edge_query_index_0'].append(d.edge_query_index_0+idx)
        data_dict['edge_query_index_1'].append(d.edge_query_index_1+compose_pos_cusum)
        data_dict['pos_query_knn_edge_idx_0'].append(d.pos_query_knn_edge_idx_0+idx)
        data_dict['pos_query_knn_edge_idx_1'].append(d.pos_query_knn_edge_idx_1+compose_pos_cusum)
        data_dict['pos_fake'].append(d.pos_fake)
        data_dict['pos_fake_knn_edge_idx_0'].append(d.pos_fake_knn_edge_idx_0+pos_fake_cusum)
        data_dict['pos_fake_knn_edge_idx_1'].append(d.pos_fake_knn_edge_idx_0+compose_pos_cusum)
        data_dict['pos_real'].append(d.pos_real)
        data_dict['pos_real_knn_edge_idx_0'].append(d.pos_real_knn_edge_idx_0+pos_real_cusum)
        data_dict['pos_real_knn_edge_idx_1'].append(d.pos_real_knn_edge_idx_0+compose_pos_cusum)
        data_dict['index_real_cps_edge_for_atten'].append(d.index_real_cps_edge_for_atten+edge_query_index_cusum)
        data_dict['tri_edge_index'].append(d.tri_edge_index+compose_pos_cusum)
        data_dict['tri_edge_feat'].append(d.tri_edge_feat)
        #data_dict['ligand_ctx_step'].append(torch.zeros_like(d.context_idx) + idx)
        data_dict['step_batch'].append(d.step_batch + idx)
        compose_pos_cusum += d.cpx_pos.size(0)
        edge_query_index_cusum += d.edge_query_index_0.size(0)
        pos_fake_cusum += d.pos_fake.size(0)
        pos_real_cusum += d.pos_real.size(0)
    data_dict['cpx_pos'] = torch.cat(data_dict['cpx_pos'])
    data_dict['cpx_feature'] = torch.cat(data_dict['cpx_feature'])
    data_dict['idx_ligand_ctx_in_cpx'] = torch.cat(data_dict['idx_ligand_ctx_in_cpx'])
    data_dict['idx_protein_in_cpx'] = torch.cat(data_dict['idx_protein_in_cpx'])
    data_dict['candidate_focal_idx_in_protein'] = torch.cat(data_dict['candidate_focal_idx_in_protein'])
    data_dict['candidate_focal_label_in_protein'] = torch.cat(data_dict['candidate_focal_label_in_protein'])
    data_dict['focal_idx_in_context'] = torch.cat(data_dict['focal_idx_in_context'])
    data_dict['focal_idx_in_context_'] = torch.cat(data_dict['focal_idx_in_context_'])
    data_dict['cpx_edge_index'] = torch.cat(data_dict['cpx_edge_index'], dim=1)
    data_dict['cpx_edge_type'] = torch.cat(data_dict['cpx_edge_type'])
    data_dict['cpx_edge_feature'] = torch.cat(data_dict['cpx_edge_feature'])
    data_dict['cpx_backbone_index'] = torch.cat(data_dict['cpx_backbone_index'])
    data_dict['focal_label'] = torch.cat(data_dict['focal_label'])
    data_dict['ligand_frontier'] = torch.cat(data_dict['ligand_frontier'])
    data_dict['y_pos'] = torch.cat(data_dict['y_pos'], dim=0)
    data_dict['edge_label'] = torch.cat(data_dict['edge_label'])
    data_dict['atom_label'] = torch.cat(data_dict['atom_label'])
    data_dict['edge_query_index_0'] = torch.cat(data_dict['edge_query_index_0'])
    data_dict['edge_query_index_1'] = torch.cat(data_dict['edge_query_index_1'])
    data_dict['pos_query_knn_edge_idx_0'] = torch.cat(data_dict['pos_query_knn_edge_idx_0'])
    data_dict['pos_query_knn_edge_idx_1'] = torch.cat(data_dict['pos_query_knn_edge_idx_1'])
    data_dict['pos_fake'] = torch.cat(data_dict['pos_fake'], dim=0)
    data_dict['pos_fake_knn_edge_idx_0'] = torch.cat(data_dict['pos_fake_knn_edge_idx_0'])
    data_dict['pos_fake_knn_edge_idx_1'] = torch.cat(data_dict['pos_fake_knn_edge_idx_1'])
    data_dict['pos_real'] = torch.cat(data_dict['pos_real'], dim=0)
    data_dict['pos_real_knn_edge_idx_0'] = torch.cat(data_dict['pos_real_knn_edge_idx_0'])
    data_dict['pos_real_knn_edge_idx_1'] = torch.cat(data_dict['pos_real_knn_edge_idx_1'])
    data_dict['index_real_cps_edge_for_atten'] = torch.cat(data_dict['index_real_cps_edge_for_atten'],dim=1)
    data_dict['tri_edge_index'] = torch.cat(data_dict['tri_edge_index'], dim=1)
    data_dict['tri_edge_feat'] = torch.cat(data_dict['tri_edge_feat'], dim=0)
    data_dict['apo_protein_idx'] = torch.cat(data_dict['apo_protein_idx']) # 
    data_dict['step_batch'] = torch.cat(data_dict['step_batch'])
    #data_dict['protein_filename'] = d.protein_filename
    #data_dict['ligand_filename'] = d.ligand_filename
    # get batch
    data_dict['cpx_pos_batch'] = torch.zeros(data_dict['cpx_pos'].size(0), dtype=torch.long)
    data_dict['y_pos_batch'] = torch.zeros(data_dict['y_pos'].size(0), dtype=torch.long)
    data_dict['edge_label_batch'] = torch.zeros(data_dict['edge_label'].size(0), dtype=torch.long)
    data_dict['atom_label_batch'] = torch.zeros(data_dict['atom_label'].size(0), dtype=torch.long)
    data = ComplexData.from_dict(data_dict)
    data.is_traj = True
    del data_dict
    return data

        