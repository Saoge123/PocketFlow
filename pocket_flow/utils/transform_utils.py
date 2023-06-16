import torch
import torch.nn.functional as F
from torch_geometric.nn.pool import knn_graph, radius_graph
from torch_geometric.transforms import Compose
from torch_geometric.utils.subgraph import subgraph
from torch_geometric.nn import knn, radius
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add
import numpy as np
import random
import copy


def count_neighbors(edge_index, symmetry, valence=None, num_nodes=None):
    assert symmetry == True, 'Only support symmetrical edges.'

    if num_nodes is None:
        num_nodes = maybe_num_nodes(edge_index)

    if valence is None:
        valence = torch.ones([edge_index.size(1)], device=edge_index.device)
    valence = valence.view(edge_index.size(1))
    return scatter_add(valence, index=edge_index[0], dim=0, dim_size=num_nodes).long()


def change_features_of_neigh(ligand_feature_full, new_num_neigh, new_num_valence, ligand_atom_num_bonds, num_atom_type=7):
    idx_n_neigh = num_atom_type + 1
    idx_n_valence = idx_n_neigh + 1
    idx_n_bonds = idx_n_valence + 1
    ligand_feature_full[:, idx_n_neigh] = new_num_neigh.long()
    ligand_feature_full[:, idx_n_valence] = new_num_valence.long()
    ligand_feature_full[:, idx_n_bonds:idx_n_bonds+3] = ligand_atom_num_bonds.long()
    return ligand_feature_full


def get_rfs_perm(nbh_list, ring_info):
    num_nodes = len(nbh_list)
    node0 = random.randint(0, num_nodes-1)
    queue,order,not_ring_queue,edge_index = [],[],[],[]
    if (ring_info[node0]>0).sum():
        queue.append(node0)
    else:
        not_ring_queue.append(node0)
    while queue or not_ring_queue:
        if queue:
            v = queue.pop()
            order.append(v)
        elif not_ring_queue:
            v = not_ring_queue.pop()
            order.append(v)
        adj_in_ring, adj_not_ring, edge_idx_step = [], [], []
        for nbh in nbh_list[v]:
            if (ring_info[nbh]>0).sum():
                adj_in_ring.append(nbh)
            else:
                adj_not_ring.append(nbh)
            if nbh in order:
                edge_idx_step.append((v,nbh))
                edge_idx_step.append((nbh,v))
        edge_index.append(edge_idx_step)
        if adj_not_ring:
            for w in adj_not_ring:
                if w not in order and w not in not_ring_queue:
                    not_ring_queue.append(w)
        # Preferential access to atoms in the same ring of w
        same_ring_pool = []
        if adj_in_ring:
            for w in adj_in_ring:
                if (w not in order and w not in queue):
                    if (ring_info[w] == ring_info[v]).sum()>0:
                        same_ring_pool.append(w)
                    else:
                        queue.append(w)
                elif w not in order and w in queue:
                    if (ring_info[w] == ring_info[v]).sum()>0:
                        same_ring_pool.append(w)
                        queue.remove(w)
                elif w not in order and (ring_info[w]>0).sum()>=1:
                    queue.remove(w)
                    queue.append(w)
            queue += same_ring_pool
    return torch.LongTensor(order), edge_index


def get_bfs_perm(nbh_list):
    num_nodes = len(nbh_list)
    num_neighbors = torch.LongTensor([len(nbh_list[i]) for i in range(num_nodes)])
    bfs_queue = [random.randint(0, num_nodes-1)]
    bfs_perm, edge_index = [], []
    num_remains = [num_neighbors.clone()]
    bfs_next_list = {}
    visited = {bfs_queue[0]}
    num_nbh_remain = num_neighbors.clone()
    while len(bfs_queue) > 0:
        current = bfs_queue.pop(0) # Remove and return item at index (default last)
        for nbh in nbh_list[current]:
            num_nbh_remain[nbh] -= 1
        bfs_perm.append(current)
        num_remains.append(num_nbh_remain.clone())
        next_candid, edge_idx_step = [], []
        for nxt in nbh_list[current]:
            if nxt in visited: continue
            next_candid.append(nxt)
            visited.add(nxt)
            for adj in nbh_list[nxt]:
                if adj in bfs_perm:
                    edge_idx_step.append((adj,nxt))
                    edge_idx_step.append((nxt,adj))
        edge_index.append(edge_idx_step)
        random.shuffle(next_candid)
        bfs_queue += next_candid
        bfs_next_list[current] = copy.copy(bfs_queue)
    return torch.LongTensor(bfs_perm), edge_index


def mask_node(data, context_idx, masked_idx, num_atom_type=10, y_pos_std=0.05):
    data.context_idx = context_idx  # for change bond index
    data.masked_idx = masked_idx
    # masked ligand atom element/feature/pos.
    data.ligand_masked_element = data.ligand_element[masked_idx]
    # data.ligand_masked_feature = data.ligand_atom_feature[masked_idx]   # For Prediction. these features are chem properties
    data.ligand_masked_pos = data.ligand_pos[masked_idx]

    # context ligand atom elment/full features/pos. Note: num_neigh and num_valence features should be changed
    data.ligand_context_element = data.ligand_element[context_idx]
    data.ligand_context_feature_full = data.ligand_atom_feature_full[context_idx]   # For Input
    data.ligand_context_pos = data.ligand_pos[context_idx]

    # new bond with ligand context atoms
    if data.ligand_bond_index.size(1) != 0:
        data.ligand_context_bond_index, data.ligand_context_bond_type = subgraph(
            context_idx,
            data.ligand_bond_index,
            edge_attr = data.ligand_bond_type,
            relabel_nodes = True,
        )
    else:
        data.ligand_context_bond_index = torch.empty([2, 0], dtype=torch.long)
        data.ligand_context_bond_type = torch.empty([0], dtype=torch.long)
    # re-calculate atom features that relate to bond
    data.ligand_context_num_neighbors = count_neighbors(
        data.ligand_context_bond_index,
        symmetry=True,
        num_nodes = context_idx.size(0),
    )
    data.ligand_context_valence = count_neighbors(
        data.ligand_context_bond_index,
        symmetry=True,
        valence=data.ligand_context_bond_type,
        num_nodes=context_idx.size(0)
    )
    data.ligand_context_num_bonds = torch.stack([
        count_neighbors(
            data.ligand_context_bond_index,
            symmetry=True,
            valence=data.ligand_context_bond_type == i,
            num_nodes=context_idx.size(0),
        ) for i in [1, 2, 3]
    ], dim = -1)
    # re-calculate ligand_context_featrure_full
    data.ligand_context_feature_full = change_features_of_neigh(
        data.ligand_context_feature_full,
        data.ligand_context_num_neighbors,
        data.ligand_context_valence,
        data.ligand_context_num_bonds,
        num_atom_type=num_atom_type
    )
    if data.ligand_masked_pos.size(0) == 0:
        data.y_pos = torch.empty([0,3], dtype=torch.float32)
    else:
        data.y_pos = data.ligand_masked_pos[0].view(-1,3)
        #data.y_pos = data.y_pos.repeat(5, 1)
        data.y_pos += torch.randn_like(data.y_pos) * y_pos_std
    data.ligand_frontier = (data.ligand_context_num_neighbors < data.ligand_num_neighbors[context_idx])
    return data


def make_pos_label(data, num_real_pos=5, num_fake_pos=5, pos_real_std=0.05, pos_fake_std=2.0, k=16):
    ligand_context_pos = data.ligand_context_pos
    ligand_masked_pos = data.ligand_masked_pos
    protein_pos = data.protein_pos
    
    if ligand_context_pos.size(0) == 0:
        # fake position
        fake_mode = protein_pos[data.candidate_focal_label_in_protein]
        p = np.ones(fake_mode.size(0), dtype=np.float32)/fake_mode.size(0)
        pos_fake_idx = np.random.choice(np.arange(fake_mode.size(0)), size=num_fake_pos, p=p)
        pos_fake = fake_mode[pos_fake_idx]
        pos_fake += torch.randn_like(pos_fake) * pos_fake_std / 2.
        # real position
        p = np.ones(ligand_masked_pos.size(0), dtype=np.float32)/ligand_masked_pos.size(0)
        pos_real_idx = np.random.choice(np.arange(ligand_masked_pos.size(0)), size=num_real_pos, p=p)
        pos_real = ligand_masked_pos[pos_real_idx]
        pos_real += torch.randn_like(pos_real) * pos_real_std
    else:
        # fake position
        fake_mode = ligand_context_pos[data.ligand_frontier]
        p = np.ones(fake_mode.size(0), dtype=np.float32)/fake_mode.size(0)
        pos_fake_idx = np.random.choice(np.arange(fake_mode.size(0)), size=num_fake_pos, p=p)
        pos_fake = fake_mode[pos_fake_idx]
        pos_fake += torch.randn_like(pos_fake) * pos_fake_std / 2.
        # real position
        p = np.ones(ligand_masked_pos.size(0), dtype=np.float32)/ligand_masked_pos.size(0)
        pos_real_idx = np.random.choice(np.arange(ligand_masked_pos.size(0)), size=num_real_pos-1, p=p)
        pos_real = ligand_masked_pos[pos_real_idx]
        pos_real += torch.randn_like(pos_real) * pos_real_std
        pos_real = torch.cat([pos_real, data.y_pos], dim=0)

    data.pos_fake = pos_fake
    pos_fake_knn_edge_idx = knn(x=data.cpx_pos, y=pos_fake, k=k, num_workers=16)
    data.pos_fake_knn_edge_idx_0, data.pos_fake_knn_edge_idx_1 = pos_fake_knn_edge_idx

    data.pos_real = pos_real
    pos_real_knn_edge_idx = knn(x=data.cpx_pos, y=pos_real, k=k, num_workers=16)
    data.pos_real_knn_edge_idx_0, data.pos_real_knn_edge_idx_1 = pos_real_knn_edge_idx
    return data


def get_complex_graph(data, len_ligand_ctx, len_compose, num_workers=1, graph_type='knn', knn=16, radius=10.0):
    
    data.cpx_edge_index = knn_graph(data.cpx_pos, knn, flow='target_to_source', num_workers=num_workers)
    # compose_knn_edge_index
    id_cpx_edge = data.cpx_edge_index[0, :len_ligand_ctx*knn] * len_compose + data.cpx_edge_index[1, :len_ligand_ctx*knn]
    # data.cpx_edge_index[0, :len_ligand_ctx*knn]
    id_ligand_ctx_edge = data.ligand_context_bond_index[0] * len_compose + data.ligand_context_bond_index[1]
    idx_edge = [torch.nonzero(id_cpx_edge == id_) for id_ in id_ligand_ctx_edge]
    # torch.nonzero(id_ligand_ctx_edge.unsqueeze(1) == id_cpx_edge.unsqueeze(0))[:,1]
    idx_edge = torch.tensor([a.squeeze() if len(a) > 0 else torch.tensor(-1) for a in idx_edge], dtype=torch.long)
    data.cpx_edge_type = torch.zeros(len(data.cpx_edge_index[0]), dtype=torch.long)  # for encoder edge embedding
    data.cpx_edge_type[idx_edge[idx_edge>=0]] = data.ligand_context_bond_type[idx_edge>=0]
    data.cpx_edge_feature = torch.cat([
        torch.ones([len(data.cpx_edge_index[0]), 1], dtype=torch.long),
        torch.zeros([len(data.cpx_edge_index[0]), 3], dtype=torch.long),
    ], dim=-1)
    data.cpx_edge_feature[idx_edge[idx_edge>=0]] = F.one_hot(data.ligand_context_bond_type[idx_edge>=0], num_classes=4)    # 0 (1,2,3)-onehot
    return data

def get_knn_graph(pos, k=16, edge_feat=None, edge_feat_index=None, num_workers=8, graph_type='knn', radius=5.5):
    if graph_type == 'rad':
        cpx_edge_index = radius_graph(pos, radius, flow='target_to_source', num_workers=num_workers)
    if graph_type == 'knn':
        cpx_edge_index = knn_graph(pos, k, flow='target_to_source', num_workers=num_workers).long()
    if isinstance(edge_feat, torch.Tensor) and isinstance(edge_feat_index, torch.Tensor):
        adj_feat_mat = torch.zeros([pos.size(0), pos.size(0)], dtype=torch.long)
        adj_feat_mat[edge_feat_index[0],edge_feat_index[1]] = edge_feat
        cpx_edge_type = adj_feat_mat[cpx_edge_index[0],cpx_edge_index[1]]
    else:
        cpx_edge_type = None
    return cpx_edge_index, cpx_edge_type

def get_complex_graph_(data, knn=16, num_workers=8, graph_type='knn', radius=5.5):
    edge_feat = torch.cat([data.ligand_context_bond_type, data.protein_bond_type]).long()
    edge_feat_index = torch.cat(
        [data.ligand_context_bond_index, data.protein_bond_index+data.ligand_context_pos.size(0)], dim=1
        ).long()
    knn_edge_index, knn_edge_type = get_knn_graph(
        data.cpx_pos, k=knn, edge_feat=edge_feat, edge_feat_index=edge_feat_index, 
        graph_type=graph_type, num_workers=num_workers, radius=radius
        )
    data.cpx_edge_index = knn_edge_index
    data.cpx_edge_type = knn_edge_type
    data.cpx_edge_feature = F.one_hot(knn_edge_type, num_classes=4)
    return data

def sample_edge_with_radius(data, r=4.0):
    y_pos = data.y_pos
    ligand_context_pos = data.ligand_context_pos
    context_idx = data.context_idx
    masked_idx = data.masked_idx
    ligand_bond_index = data.ligand_bond_index
    ligand_bond_type = data.ligand_bond_type
    # select the atoms whose distance < r between pos_query as edge samples
    edge_index_radius = radius(ligand_context_pos, y_pos, r=r, num_workers=16)
    # get the labels of edge samples
    mask = [i in masked_idx[0] and j in context_idx[edge_index_radius[1]] for i,j in zip(*ligand_bond_index)]
    new_idx_1 = torch.nonzero((ligand_bond_index[:,mask][1].view(-1,1) == context_idx).any(0)).view(-1)
    real_bond_type_in_edge_index_radius = torch.nonzero((new_idx_1.view(-1,1) == edge_index_radius[1]).any(0)).view(-1)
    edge_label = torch.zeros(edge_index_radius.size(1), dtype=torch.long)
    edge_label[real_bond_type_in_edge_index_radius] = ligand_bond_type[mask]
    data.edge_query_index_0, data.edge_query_index_1 = edge_index_radius
    data.edge_label = edge_label
    return data

def get_tri_edges(edge_index_query, pos_query, idx_ligand, ligand_bond_index, ligand_bond_type):
    row, col = edge_index_query
    acc_num_edges = 0
    index_real_cps_edge_i_list, index_real_cps_edge_j_list = [], []  # index of real-ctx edge (for attention)
    for node in torch.arange(pos_query.size(0)):
        num_edges = (row == node).sum()
        index_edge_i = torch.arange(num_edges, dtype=torch.long, ) + acc_num_edges
        index_edge_i, index_edge_j = torch.meshgrid(index_edge_i, index_edge_i, indexing=None)
        index_edge_i, index_edge_j = index_edge_i.flatten(), index_edge_j.flatten()
        index_real_cps_edge_i_list.append(index_edge_i)
        index_real_cps_edge_j_list.append(index_edge_j)
        acc_num_edges += num_edges
    index_real_cps_edge_i = torch.cat(index_real_cps_edge_i_list, dim=0).to(pos_query.device)  # add len(real_compose_edge_index) in the dataloader for batch
    index_real_cps_edge_j = torch.cat(index_real_cps_edge_j_list, dim=0).to(pos_query.device)

    node_a_cps_tri_edge = col[index_real_cps_edge_i]  # the node of tirangle edge for the edge attention (in the compose)
    node_b_cps_tri_edge = col[index_real_cps_edge_j]
    n_context = len(idx_ligand)
    adj_mat = (torch.zeros([n_context, n_context], dtype=torch.long) - torch.eye(n_context, dtype=torch.long))
    adj_mat = adj_mat.to(ligand_bond_index.device)
    adj_mat[ligand_bond_index[0], ligand_bond_index[1]] = ligand_bond_type
    tri_edge_type = adj_mat[node_a_cps_tri_edge, node_b_cps_tri_edge]
    tri_edge_feat = (tri_edge_type.view([-1, 1]) == torch.tensor([[-1, 0, 1, 2, 3]]).to(tri_edge_type.device)).long()

    index_real_cps_edge_for_atten = torch.stack([
        index_real_cps_edge_i, index_real_cps_edge_j  # plus len(real_compose_edge_index_0) for dataloader batch
    ], dim=0)
    tri_edge_index = torch.stack([
        node_a_cps_tri_edge, node_b_cps_tri_edge  # plus len(compose_pos) for dataloader batch
    ], dim=0)
    return index_real_cps_edge_for_atten, tri_edge_index, tri_edge_feat