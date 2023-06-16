import torch
import numpy as np
# from torch_geometric.data import Data, DataLoader, Batch
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader

FOLLOW_BATCH = [] #['protein_element', 'ligand_context_element', 'pos_real', 'pos_fake']


class ComplexData(Data):

    def __init__(self, *args, **kwargs):
        super(ComplexData, self).__init__(*args, **kwargs)
        self.is_traj = False

    @staticmethod
    def from_protein_ligand_dicts(protein_dict=None, ligand_dict=None, **kwargs):
        instance = ComplexData(**kwargs)

        if protein_dict is not None:
            for key, item in protein_dict.items():
                instance['protein_' + key] = item

        if ligand_dict is not None:
            for key, item in ligand_dict.items():
                instance['ligand_' + key] = item

        instance['ligand_nbh_list'] = {i.item():[j.item() for k, j in enumerate(instance.ligand_bond_index[1]) \
                                                 if instance.ligand_bond_index[0, k].item() == i] \
                                                 for i in instance.ligand_bond_index[0]}
        return instance

    def __inc__(self, key, value, *args, **kwargs):
        KEYS = {
            'idx_ligand_ctx_in_cpx', 'idx_protein_in_cpx', 'focal_idx_in_cpx', 'focal_idx_in_context_',
            'focal_idx_in_context', 'cpx_knn_edge_index', 'edge_query_index_1', 'pos_query_edge_idx_1',
            'pos_fake_knn_edge_idx_1', 'pos_real_knn_edge_idx_1', 'tri_edge_index', 'apo_protein_idx',
            'candidate_focal_idx_in_protein', 'cpx_backbone_index'
            }
        if key in KEYS:
            return self['cpx_pos'].size(0)
        elif key == 'edge_query_index_0':
            return self['y_pos'].size(0)
        elif key == 'pos_query_edge_idx_0':
            return self['y_pos'].size(0)
        elif key == 'pos_fake_knn_edge_idx_0':
            return self['pos_fake'].size(0)
        elif key == 'pos_real_knn_edge_idx_0':
            return self['pos_real'].size(0)
        elif key == 'step_batch':
            return self['step_batch'].max() + 1
        elif key == 'ligand_bond_index':
            return self['ligand_element'].size()
        elif key == 'index_real_cps_edge_for_atten':
            return self['edge_query_index_0'].size(0)
        else:
            return super().__inc__(key, value)
        '''if key == 'idx_ligand_ctx_in_cpx':
            return self['cpx_pos'].size(0)
        elif key == 'idx_protein_in_cpx':
            return self['cpx_pos'].size(0)
        elif key == 'focal_idx_in_cpx':
            return self['cpx_pos'].size(0)
        elif key == 'focal_idx_in_context_':
            return self['cpx_pos'].size(0)
        elif key == 'focal_idx_in_context':
            return self['cpx_pos'].size(0)
        elif key == 'cpx_knn_edge_index':
            return self['cpx_pos'].size(0)
        elif key == 'edge_query_index_0':
            return self['y_pos'].size(0)
        elif key == 'edge_query_index_1':
            return self['cpx_pos'].size(0)
        elif key == 'pos_query_edge_idx_0':
            return self['y_pos'].size(0)
        elif key == 'pos_query_edge_idx_1':
            return self['cpx_pos'].size(0)
        elif key == 'pos_fake_knn_edge_idx_0':
            return self['pos_fake'].size(0)
        elif key == 'pos_fake_knn_edge_idx_1':
            return self['cpx_pos'].size(0)
        elif key == 'pos_real_knn_edge_idx_0':
            return self['pos_real'].size(0)
        elif key == 'pos_real_knn_edge_idx_1':
            return self['cpx_pos'].size(0)
        elif key == 'tri_edge_index':
            return self['cpx_pos'].size(0)
        elif key == 'apo_protein_idx':
            return self['cpx_pos'].size(0)
        elif key == 'candidate_focal_idx_in_protein':
            return self['cpx_pos'].size(0)
        elif key == 'step_batch':
            return self['step_batch'].max() + 1
        elif key == 'ligand_bond_index':
            return self['ligand_element'].size()
        elif key == 'index_real_cps_edge_for_atten':
            return self['edge_query_index_0'].size(0)
        else:
            return super().__inc__(key, value)'''
    @property
    def num_nodes(self):
        if self.is_traj:
            return self.cpx_pos.size(0)
        else:
            return self.protein_pos.size(0) + self.context_idx.size(0)


class ProteinLigandDataLoader(DataLoader):

    def __init__(
        self, 
        dataset, 
        batch_size = 1, 
        shuffle = False, 
        follow_batch = ['ligand_element', 'protein_element'], 
        **kwargs
    ):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, follow_batch=follow_batch, **kwargs)


def batch_from_data_list(data_list):
    return Batch.from_data_list(data_list, follow_batch=['ligand_element', 'protein_element'])


def torchify_dict(data):
    output = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            output[k] = torch.from_numpy(v)
        else:
            output[k] = v
    return output

    