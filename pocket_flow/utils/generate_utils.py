import torch
import torch.nn.functional as F
from rdkit import Chem, Geometry
from rdkit.Chem import Descriptors
import numpy as np
import copy
import itertools


max_valence_dict = {
                6:torch.LongTensor([4]), 7:torch.LongTensor([3]), 8:torch.LongTensor([2]),
                9:torch.LongTensor([1]), 15:torch.LongTensor([5]), 16:torch.LongTensor([6]),
                17:torch.LongTensor([1]), 35:torch.LongTensor([1]), 53:torch.LongTensor([1]),
                1:torch.LongTensor([1])
                }
def add_ligand_atom_to_data(
    old_data, pos, element, bond_index, bond_type, type_map=[1,6,7,8,9,15,16,17,35,53],
    max_valence_dict=max_valence_dict):
    data = old_data.clone()
    if 'max_atom_valence' not in data.__dict__['_store']:
        data.max_atom_valence = torch.empty(0, dtype=torch.long)
    # add position of new atom to context
    data.ligand_context_pos = torch.cat([
        data.ligand_context_pos, pos.view(1, 3).to(data.ligand_context_pos)
    ], dim=0)

    # add feature of new atom to context
    data.ligand_context_feature_full = torch.cat([
        data.ligand_context_feature_full,
        torch.cat([
            F.one_hot(element.view(1), len(type_map)).to(data.ligand_context_feature_full), # (1, num_elements)
            torch.tensor([[1, 0, 0]]).to(data.ligand_context_feature_full),  # is_mol_atom, num_neigh (placeholder), valence (placeholder)
            torch.tensor([[0, 0, 0]]).to(data.ligand_context_feature_full)  # num_of_bonds 1, 2, 3(placeholder)
        ], dim=1)
    ], dim=0)
    data.context_idx = torch.arange(data.context_idx.size(0)+1)
    # 
    idx_num_neigh = len(type_map) + 1
    idx_valence = idx_num_neigh + 1
    idx_num_of_bonds = idx_valence + 1

    # add type of new atom to context
    element = torch.LongTensor([type_map[element.item()]])
    data.ligand_context_element = torch.cat([
        data.ligand_context_element, element.view(1).to(data.ligand_context_element)
    ])
    max_new_atom_valence = max_valence_dict[element.item()].to(data.max_atom_valence)
    data.max_atom_valence = torch.cat([data.max_atom_valence, max_new_atom_valence])
    
    # change the feature of new atom to context according to ligand context
    if len(bond_type) != 0:
        bond_index, bond_type = remove_triangle(
            pos, data.ligand_context_pos, data.ligand_context_bond_index, data.ligand_context_bond_type, 
                    bond_index, bond_type)
        bond_index[0, :] = len(data.ligand_context_pos) - 1
        '''bond_type = check_double_bond(
            data.ligand_context_bond_index, 
            data.ligand_context_bond_type, 
            bond_index, 
            bond_type
            )'''
        bond_type = check_valence_is_2(
                            bond_index, bond_type,
                            data.ligand_context_element, data.ligand_context_valence
                            )
        bond_vec = data.ligand_context_pos[bond_index[0]] - data.ligand_context_pos[bond_index[1]]
        bond_lengths = torch.norm(bond_vec, dim=-1, p=2)
        if (bond_lengths > 3).any():
            print(bond_lengths)
        
        bond_index_all = torch.cat([bond_index, torch.stack([bond_index[1, :], bond_index[0, :]], dim=0)], dim=1)
        bond_type_all = torch.cat([bond_type, bond_type], dim=0)

        data.ligand_context_bond_index = torch.cat([
            data.ligand_context_bond_index, bond_index_all.to(data.ligand_context_bond_index)
        ], dim=1)

        data.ligand_context_bond_type = torch.cat([
            data.ligand_context_bond_type,
            bond_type_all
        ])
        # modify atom features related to bonds
        # previous atom
        data.ligand_context_feature_full[bond_index[1, :], idx_num_neigh] += 1 # num of neigh of previous nodes
        data.ligand_context_feature_full[bond_index[1, :], idx_valence] += bond_type # valence of previous nodes
        data.ligand_context_feature_full[bond_index[1, :], idx_num_of_bonds + bond_type - 1] += 1  # num of bonds of 
        # the new atom
        data.ligand_context_feature_full[-1, idx_num_neigh] += len(bond_index[1]) # num of neigh of last node
        data.ligand_context_feature_full[-1, idx_valence] += torch.sum(bond_type) # valence of last node
        for bond in [1, 2, 3]:
            data.ligand_context_feature_full[-1, idx_num_of_bonds + bond - 1] += (bond_type == bond).sum()  # num of bonds of last node
    data.ligand_context_valence = data.ligand_context_feature_full[:,idx_valence]
    del old_data
    return data


def data2mol(data, raise_error=True, sanitize=True):
    element = data.ligand_context_element.clone().cpu().tolist()
    bond_index = data.ligand_context_bond_index.clone().cpu().tolist()
    bond_type = data.ligand_context_bond_type.clone().cpu().tolist()
    pos = data.ligand_context_pos.clone().cpu().tolist()
    n_atoms = len(pos)
    rd_mol = Chem.RWMol()
    rd_conf = Chem.Conformer(n_atoms)

    # add atoms and coordinates
    for i, atom in enumerate(element):
        rd_atom = Chem.Atom(atom)
        rd_mol.AddAtom(rd_atom)
        rd_coords = Geometry.Point3D(*pos[i])
        rd_conf.SetAtomPosition(i, rd_coords)
    rd_mol.AddConformer(rd_conf)
    # add atoms and coordinates
    # add atoms and coordinates
    for i, type_this in enumerate(bond_type):
        node_i, node_j = bond_index[0][i], bond_index[1][i]
        if node_i < node_j:
            if type_this == 1:
                rd_mol.AddBond(node_i, node_j, Chem.BondType.SINGLE)
            elif type_this == 2:
                rd_mol.AddBond(node_i, node_j, Chem.BondType.DOUBLE)
            elif type_this == 3:
                rd_mol.AddBond(node_i, node_j, Chem.BondType.TRIPLE)
            elif type_this == 12:
                rd_mol.AddBond(node_i, node_j, Chem.BondType.AROMATIC)
            else:
                raise Exception('unknown bond order {}'.format(type_this))
    # modify
    try:
        rd_mol = modify_submol(rd_mol)
    except:
        if raise_error:
            raise MolReconsError()
        else:
            print('MolReconsError')
    # check valid
    rd_mol_check = Chem.MolFromSmiles(Chem.MolToSmiles(rd_mol))
    if rd_mol_check is None:
        if raise_error:
            raise MolReconsError()
        else:
            print('MolReconsError')
    rd_mol = rd_mol.GetMol()
    if 12 in bond_type:  # mol may directlu come from ture mols and contains aromatic bonds
        Chem.Kekulize(rd_mol, clearAromaticFlags=True)
    if sanitize:
        Chem.SanitizeMol(rd_mol, Chem.SANITIZE_ALL^Chem.SANITIZE_KEKULIZE^Chem.SANITIZE_SETAROMATICITY)
    #rd_mol = modify(rd_mol)
    return rd_mol


def modify_submol(mol):  # modify mols containing C=N(C)O
    submol = Chem.MolFromSmiles('C=N(C)O', sanitize=False)
    sub_fragments = mol.GetSubstructMatches(submol)
    for fragment in sub_fragments:
        atomic_nums = np.array([mol.GetAtomWithIdx(atom).GetAtomicNum() for atom in fragment])
        idx_atom_N = fragment[np.where(atomic_nums == 7)[0][0]]
        idx_atom_O = fragment[np.where(atomic_nums == 8)[0][0]]
        mol.GetAtomWithIdx(idx_atom_N).SetFormalCharge(1)  # set N to N+
        mol.GetAtomWithIdx(idx_atom_O).SetFormalCharge(-1)  # set O to O-
    return mol


class MolReconsError(Exception):
    pass

def add_context(data):
    data.ligand_context_pos = data.ligand_pos
    data.ligand_context_element = data.ligand_element
    data.ligand_context_bond_index = data.ligand_bond_index
    data.ligand_context_bond_type = data.ligand_bond_type
    return data


def check_valency(mol):
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True
    except ValueError:
        return False


def check_double_bond(
    ligand_context_bond_index, ligand_context_bond_type, bond_index_to_add, bond_type_to_add
    ):
    bond_type_in_place = [
        ligand_context_bond_type[ix==ligand_context_bond_index[0]]\
             for ix in bond_index_to_add[1]
             ]
    has_double = torch.BoolTensor([(bt==2).sum()>0 for bt in bond_type_in_place])
    bond_type_to_add_has_double = bond_type_to_add >= 2
    mask = torch.stack([has_double,bond_type_to_add_has_double]).all(dim=0)
    bond_type_to_add = bond_type_to_add - mask.long()
    return bond_type_to_add


def check_valence_is_2(
    bond_index_to_add, bond_type_to_add,
    ligand_context_element, ligand_context_valence
    ):
    atom_type_to_add = ligand_context_element[-1]
    new_atom_type_is_C = (atom_type_to_add == 6).view(-1)
    if new_atom_type_is_C:
        atom_valence_in_place = ligand_context_valence[bond_index_to_add[1]] >= 2
        atom_type_in_place_is_C = ligand_context_element[bond_index_to_add[1]] == 6
        bond_type_to_add_mask = bond_type_to_add >= 2

        mask = torch.stack([atom_valence_in_place, atom_type_in_place_is_C, bond_type_to_add_mask]).all(dim=0)
        bond_type_to_add = bond_type_to_add - mask.long()
    return bond_type_to_add
    

def remove_triangle(pos_to_add, ligand_context_pos, ligand_context_bond_index, ligand_context_bond_type, 
                    bond_index_to_add, bond_type_to_add):
    new_j = bond_index_to_add[1]
    atom_in_place_adjs = [
        ligand_context_bond_index[1][ligand_context_bond_index[0]==i] for i in new_j
        ]
    L = []
    for j in new_j:
        l = []
        for i in atom_in_place_adjs:
            if j in i:
                l.append(True)
            else:
                l.append(False)
        L.append(l)
    adj_mask = torch.LongTensor(L).any(-1)
    if adj_mask.sum() > 0:
        dist = torch.norm(pos_to_add.view(-1,3) - ligand_context_pos[new_j], dim=-1, p=2)
        dist_mask_idx = torch.nonzero(adj_mask).view(-1)
        max_dist_idx_to_remove = dist_mask_idx[dist[dist_mask_idx].argmax()]
        mask = (torch.arange(len(bond_type_to_add)) == max_dist_idx_to_remove) == False
        bond_index_to_add = bond_index_to_add[:, mask]
        bond_type_to_add = bond_type_to_add[mask]
    return bond_index_to_add, bond_type_to_add


PATTERNS = [
        #Chem.MolFromSmarts('[C,N]1~&@[C,N]~&@[C,N]~&@[C,N]~&@[C,N]~&@[C,N]~&@1'), # '[R]1~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@1'
        Chem.MolFromSmarts('[N]1~&@[N]~&@[C]~&@[C]~&@[C]~&@[C]~&@1'),   # '[R]1~&@[R]~&@[R]~&@[R]~&@[R]~&@1'
        Chem.MolFromSmarts('[N]1~&@[C]~&@[N]~&@[C]~&@[C]~&@[C]~&@1'),
        Chem.MolFromSmarts('[N]1~&@[C]~&@[C]~&@[N]~&@[C]~&@[C]~&@1'),
        Chem.MolFromSmarts('[N]1~&@[C]~&@[C]~&@[C]~&@[C]~&@[C]~&@1'),
        Chem.MolFromSmarts('[C]1~&@[C]~&@[C]~&@[C]~&@[C]~&@[C]~&@1')
    ]
PATTERNS_1 = [
        [Chem.MolFromSmarts('[#6,#7,#8]-[#7]1~&@[#6,#7]~&@[#6,#7]~&@[#6,#7]~&@[#6,#7]~&@[#6,#7]~&@1'),
        Chem.MolFromSmarts('[C,N,O]-[N]1~&@[C,N]~&@[C,N]~&@[C,N]~&@[C,N]~&@[C,N]~&@1')],
        [Chem.MolFromSmarts('[#6,#7,#8]-[#6]1(-[#6,#7,#8])~&@[#6,#7]~&@[#6,#7]~&@[#6,#7]~&@[#6,#7]~&@[#6,#7]~&@1'),
        Chem.MolFromSmarts('[C,N,O]-[C]1(-[C,N,O])~&@[C,N]~&@[C,N]~&@[C,N]~&@[C,N]~&@[C,N]~&@1')]
        #Chem.MolFromSmarts('[C,N,O]-[N]1~&@[C]~&@[C]~&@[N]~&@[C]~&@[C]-1'),
        #Chem.MolFromSmarts('[C,N,O]-[N]1~&@[C]~&@[C]~&@[C]~&@[C]~&@[C]-1'),
    ]
MAX_VALENCE = {'C':4, 'N':3}
def modify(mol, max_double_in_6ring=0):
    #atoms = mol.GetAtoms()
    mol_copy = copy.deepcopy(mol)
    mw = Chem.RWMol(mol)

    p1 = Chem.MolFromSmarts('[#6,#7]=[#6]1-[#6,#7]~&@[#6,#7]~&@[#6,#7]~&@[#6,#7]~&@[#6,#7]-1')
    p1_ = Chem.MolFromSmarts('[C,N]=[C]1-[C,N]~&@[C,N]~&@[C,N]~&@[C,N]~&@[C,N]-1')
    subs = set(list(mw.GetSubstructMatches(p1)) + list(mw.GetSubstructMatches(p1_)))
    subs_set_1 = [set(s) for s in subs]
    for sub in subs:
        comb = itertools.combinations(sub, 2)
        #b_list = [(c, mw.GetBondBetweenAtoms(*c)) for c in comb if mw.GetBondBetweenAtoms(*c) is not None]
        change_double = False
        r_b_double = 0
        b_list = []
        for ix,c in enumerate(comb):
            b = mw.GetBondBetweenAtoms(*c)
            if ix == 0:
                bt = b.GetBondType().__str__()
                is_r = b.IsInRingSize(6)
                b_list.append((c, bt, is_r))
                continue
            if b is not None:
                bt = b.GetBondType().__str__()
                is_r = b.IsInRing()
                b_list.append((c, bt, is_r))
                if is_r is True and bt == 'DOUBLE':
                    r_b_double += 1
                    if r_b_double > max_double_in_6ring:
                        change_double = True
        if change_double:
            for ix,b in enumerate(b_list):
                if ix == 0:
                    if b[-1] is False:
                        mw.RemoveBond(*b[0])
                        mw.AddBond(*b[0], Chem.rdchem.BondType.SINGLE)
                    else:
                        continue
                if b[1] == 'DOUBLE' and b[-1] is False:
                    mw.RemoveBond(*b[0])
                    mw.AddBond(*b[0], Chem.rdchem.BondType.SINGLE)
                    break
    
    #p2 = Chem.MolFromSmarts('[C,N,O]-[N]1~&@[C,N]~&@[C,N]~&@[C,N]~&@[C,N]~&@[C,N]-1')
    for p2 in PATTERNS_1:
        Chem.GetSSSR(mw)
        subs2 = set(list(mw.GetSubstructMatches(p2[0])) + list(mw.GetSubstructMatches(p2[1])))
        for sub in subs2:
            comb = itertools.combinations(sub, 2)
            b_list = [(c, mw.GetBondBetweenAtoms(*c)) for c in comb if mw.GetBondBetweenAtoms(*c) is not None]
            for b in b_list:
                if b[-1].GetBondType().__str__() == 'DOUBLE':
                    mw.RemoveBond(*b[0])
                    mw.AddBond(*b[0], Chem.rdchem.BondType.SINGLE)

    Chem.GetSSSR(mw)
    p3 = Chem.MolFromSmarts('[#8]=[#6]1-[#6,#7]~&@[#6,#7]~&@[#6,#7]~&@[#6,#7]~&@[#6,#7]-1')
    p3_ = Chem.MolFromSmarts('[O]=[C]1-[C,N]~&@[C,N]~&@[C,N]~&@[C,N]~&@[C,N]-1')
    subs = set(list(mw.GetSubstructMatches(p3)) + list(mw.GetSubstructMatches(p3_)))
    subs_set_2 = [set(s) for s in subs]
    for sub in subs:
        comb = itertools.combinations(sub, 2)
        b_list = [(c, mw.GetBondBetweenAtoms(*c)) for c in comb if mw.GetBondBetweenAtoms(*c) is not None]
        for b in b_list:
            if b[-1].GetBondType().__str__() == 'DOUBLE' and b[-1].IsInRing() is True:
                mw.RemoveBond(*b[0])
                mw.AddBond(*b[0], Chem.rdchem.BondType.SINGLE)

    p = Chem.MolFromSmarts('[#6,#7]1~&@[#6,#7]~&@[#6,#7]~&@[#6,#7]~&@[#6,#7]~&@[#6,#7]~&@1')
    p_ = Chem.MolFromSmarts('[C,N]1~&@[C,N]~&@[C,N]~&@[C,N]~&@[C,N]~&@[C,N]~&@1')
    Chem.GetSSSR(mw)
    subs = set(list(mw.GetSubstructMatches(p)) + list(mw.GetSubstructMatches(p_)))
    subs_set_3 = [set(s) for s in subs]
    for sub in subs:
        pass_sub = False
        if subs_set_2:
            for s in subs_set_2:
                if len(s-set(sub)) == 1:
                    pass_sub = True
                    break
        if pass_sub:
            continue

        bond_list = [(i,sub[0]) if ix+1==len(sub) else (i, sub[ix+1]) for ix,i in enumerate(sub)]
        if len(bond_list) == 0:
            continue
        atoms = [mw.GetAtomWithIdx(i) for i in sub]
        for a in atoms:
            if a.GetExplicitValence()==MAX_VALENCE[a.GetSymbol()] and a.GetHybridization().__str__()=='SP3':
                break
        else:
            bond_type = [mw.GetBondBetweenAtoms(*b).GetBondType().__str__() for b in bond_list]
            if bond_type.count('DOUBLE') > max_double_in_6ring:
                for b in bond_list:
                    mw.RemoveBond(*b)
                    mw.AddBond(*b, Chem.rdchem.BondType.AROMATIC)
    
    # get new mol from modified mol
    conf = mw.GetConformer()
    rd_mol = Chem.RWMol()
    rd_conf = Chem.Conformer(mw.GetNumAtoms())
    for i, atom in enumerate(mw.GetAtoms()):
        rd_atom = Chem.Atom(atom.GetAtomicNum())
        rd_mol.AddAtom(rd_atom)
        rd_coords = conf.GetAtomPosition(i)
        rd_conf.SetAtomPosition(i, rd_coords)
    rd_mol.AddConformer(rd_conf)
    
    for i, bond in enumerate(mw.GetBonds()):
        bt = bond.GetBondType()
        node_i = bond.GetBeginAtomIdx()
        node_j = bond.GetEndAtomIdx()
        rd_mol.AddBond(node_i, node_j, bt)
    out_mol = rd_mol.GetMol()

    # check validility of the new mol
    mol_check = Chem.MolFromSmiles(Chem.MolToSmiles(out_mol))
    if mol_check:
        try:
            Chem.Kekulize(out_mol)
            del mol_copy
            return out_mol
        except:
            del mol
            return mol_copy
    else:
        del mol
        return mol_copy


def save_sdf(mol_list, save_name='mol_gen.sdf'):
    writer = Chem.SDWriter(save_name)
    writer.SetProps(['LOGP', 'MW'])
    for i, mol in enumerate(mol_list):
        mw = Descriptors.ExactMolWt(mol)
        logp = Descriptors.MolLogP(mol)
        mol.SetProp('MW', '%.2f' %(mw))
        mol.SetProp('LOGP', '%.2f' %(logp))
        mol.SetProp('_Name', 'No_%s' %(i))
        Chem.Kekulize(mol)
        writer.write(mol)
    writer.close()


def check_alert_structure(mol, alert_smarts):
    Chem.GetSSSR(mol)
    pattern = Chem.MolFromSmarts(alert_smarts)
    subs = mol.GetSubstructMatches(pattern)
    if len(subs) == 0:
        return False
    else:
        return True

def check_alert_structures(mol, alert_smarts_list):
    Chem.GetSSSR(mol)
    patterns = [Chem.MolFromSmarts(sma) for sma in alert_smarts_list]
    for p in patterns:
        subs = mol.GetSubstructMatches(p)
        if len(subs) != 0:
            return True
    else:
        return False
