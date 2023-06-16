import torch
from pocket_flow.utils import Protein, Ligand, ComplexData, torchify_dict, is_in_ring
import numpy as np
from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem.rdchem import BondType
import os


class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__

ATOM_FAMILIES = ['Acceptor', 'Donor', 'Aromatic', 'Hydrophobe', 'LumpedHydrophobe', 'NegIonizable', 'PosIonizable', 'ZnBinder']
ATOM_FAMILIES_ID = {s: i for i, s in enumerate(ATOM_FAMILIES)}
BOND_TYPES = {t: i for i, t in enumerate(BondType.names.values())}
BOND_NAMES = {i: t for i, t in enumerate(BondType.names.keys())}


empty_pocket_dict = {}
empty_pocket_dict = Dict(empty_pocket_dict)
empty_pocket_dict.element = np.empty(0, dtype=np.int64)
empty_pocket_dict.pos = np.empty([0,3], dtype=np.float32)
empty_pocket_dict.is_backbone = np.empty(0, dtype=bool)
empty_pocket_dict.atom_name = []
empty_pocket_dict.atom_to_aa_type = np.empty(0, dtype=np.int64)
empty_pocket_dict.molecule_name = None
empty_pocket_dict.bond_index = np.empty([2,0], dtype=np.int64)
empty_pocket_dict.bond_type = np.empty(0, dtype=np.int64)
empty_pocket_dict.filename = None

def parse_sdf_to_dict(rdmol, fake_pokect_dict=empty_pocket_dict):
    fdefName = os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
    #rdmol = next(iter(Chem.SDMolSupplier(mol_file, removeHs=True)))
    try:
        Chem.Kekulize(rdmol)
        ring_info = is_in_ring(rdmol)
        conformer = rdmol.GetConformer()
        feat_mat = np.zeros([rdmol.GetNumAtoms(), len(ATOM_FAMILIES)], dtype=np.int64)
        for feat in factory.GetFeaturesForMol(rdmol):
            feat_mat[feat.GetAtomIds(), ATOM_FAMILIES_ID[feat.GetFamily()]] = 1
    
        element, pos, atom_mass = [], [], []
        for a in rdmol.GetAtoms():
            element.append(a.GetAtomicNum())
            pos.append(conformer.GetAtomPosition(a.GetIdx()))
            atom_mass.append(a.GetMass())
        element = np.array(element, dtype=np.int64)
        pos = np.array(pos, dtype=np.float32)
        atom_mass = np.array(atom_mass, np.float32)
        center_of_mass = (pos * atom_mass.reshape(-1,1)).sum(0)/atom_mass.sum()
        
        edge_index, edge_type = [], []
        for b in rdmol.GetBonds():
            row = [b.GetBeginAtomIdx(), b.GetEndAtomIdx()]
            col = [b.GetEndAtomIdx(), b.GetBeginAtomIdx()]
            edge_index.extend([row, col])
            edge_type.extend([BOND_TYPES[b.GetBondType()]] * 2)
        edge_index = np.array(edge_index)
        edge_index_perm = edge_index[:,0].argsort()
        edge_index = edge_index[edge_index_perm].T
        edge_type = np.array(edge_type)[edge_index_perm]
        
        ligand_dict =  {'element': element,
                'pos': pos,
                'bond_index': edge_index,
                'bond_type': edge_type,
                'center_of_mass': center_of_mass,
                'atom_feature': feat_mat,
                'ring_info': ring_info,
                'filename':None
               }
        cpx_data = ComplexData.from_protein_ligand_dicts(
                protein_dict=torchify_dict(fake_pokect_dict),
                ligand_dict=torchify_dict(ligand_dict)
                )
        cpx_data = pickle.dumps(cpx_data)
    except:
        cpx_data = None
    return cpx_data

import gzip
import glob
from tqdm.auto import tqdm
import lmdb
import pickle
from multiprocessing import Pool


sdf_path='./path/to/ZINC/dataset/'
processed_path='./pretrain_data/ZINC_PretrainingDataset.lmdb'
sdf_list = glob.glob(sdf_path+'/*.sdf')
db = lmdb.open(
            processed_path,
            map_size=200*(1024*1024*1024),   # 200GB
            create=True,
            subdir=False,
            readonly=False, # Writable
        )
index = 0
index_list = []
for ix, sdf_supplier in enumerate(tqdm(sdf_list)):
    mol_list = list(Chem.ForwardSDMolSupplier(gzip.open(sdf_supplier), removeHs=True))
    torch.multiprocessing.set_sharing_strategy('file_system')
    pool = Pool(processes=16)
    List = pool.map(parse_sdf_to_dict, mol_list)
    pool.close()
    pool.join()
    with db.begin(write=True, buffers=True) as txn:
        for data in List:
            if data is None: continue
            key = str(index).encode()
            txn.put(
                key = key,
                value = data
            )
            index_list.append(key)
            index += 1
db.close()
index_list = np.array(index_list)
np.save(processed_path.split('.')[0]+'_Keys', index_list)