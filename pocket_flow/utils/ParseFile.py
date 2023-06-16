import os
import copy
import numpy as np
from rdkit import Chem, RDConfig
from rdkit.Chem.rdchem import BondType
from rdkit.Chem import ChemicalFeatures
from .residues_base import RESIDUES_TOPO, RESIDUES_TOPO_WITH_H
try:
    import pymol
except:
    print('we can not compute the atoms on the surface of protein, '\
          'because pymol can not be imported')
    pass
#from .BaseFeatures import Mol2Graph, RESIDUE_GRAPH_WITHOUT_H, RESIDUE_GRAPH_WITH_H
#from .utils import coordinate_adjusting
#from .Data import LigandData

ATOM_TYPE_WITH_HYBIRD = [
    'SP3_C', 'SP2_C', 'SP_C', 'SP3_N', 'SP2_N', 'SP_N', 'SP3_O', 'SP2_O', 'SP3_F', 'SP3_P',
    'SP2_P', 'SP3D_P', 'SP3_S', 'SP2_S', 'SP3D_S', 'SP3D2_S', 'SP3_Cl', 'SP3_Br', 'SP3_I'
    ]
ATOM_MAP = [6, 6, 6, 7, 7, 7, 8, 8, 9, 15, 15, 15, 16, 16, 16, 16, 17, 35, 53]
PT = Chem.GetPeriodicTable()
BACKBONE_SYMBOL = {'N', 'CA', 'C', 'O'}
AMINO_ACID_TYPE = {
    'CYS':0, 'GLY':1, 'ALA':2, 'THR':3, 'LYS':4, 'PRO':5, 'VAL':6, 'SER':7, 'ASN':8, 'LEU':9,
    'GLN':10, 'MET':11, 'ASP':12, 'TRP':13, 'HIS':14, 'GLU':15, 'ARG':16, 'ILE':17, 'PHE':18,
    'TYR':19
    }


class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


class Atom(object):
    def __init__(self, atom_info):
        
        self.idx = int(atom_info[6:11])
        self.name = atom_info[12:16].strip()
        self.res_name = atom_info[17:20].strip()  # atom_info[17:20].strip()
        self.chain = atom_info[21:22].strip()
        self.res_idx = int(atom_info[22:26])
        self.coord = np.array([float(atom_info[30:38].strip()),
                               float(atom_info[38:46].strip()),
                               float(atom_info[46:54].strip())])
        self.occupancy = float(atom_info[54:60])
        self.temperature_factor = float(atom_info[60:66].strip())
        self.seg_id = atom_info[72:76].strip()
        self.element = atom_info[76:78].strip()
        if self.element == 'SE':
            self.element = 'S'
            self.name = 'SD'
            self.res_name = 'MET'
        self.mass = PT.GetAtomicWeight(self.element)
        if self.occupancy < 1.0:
            self.is_disorder = True
        else:
            self.is_disorder = False
        self.is_surf = atom_info.split()[-1] == 'surf'

    @property
    def to_string(self):
        fmt = '{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}        {:<2s}{:2s}'    # https://cupnet.net/pdb-format/
        out = fmt.format('ATOM', self.idx, self.name, '', self.res_name, 
                       self.chain, self.res_idx,'', self.coord[0],
                       self.coord[1], self.coord[2], self.occupancy,
                       self.temperature_factor, self.seg_id, self.element)
        return out
    
    @property
    def to_dict(self):
        return {
            'element': PT.GetAtomicNumber(self.element),
            'pos': self.coord,
            'is_backbone': self.name in BACKBONE_SYMBOL,
            'atom_name': self.name,
            'atom_to_aa_type': AMINO_ACID_TYPE[self.res_name]
             }
    
    def __repr__(self):
        info = 'name={}, index={}, res={}, chain={}, is_disorder={}'.format(self.name, 
                                                                 self.idx, 
                                                                 self.res_name+str(self.res_idx),
                                                                 self.chain, 
                                                                 self.is_disorder)
        return '{}({})'.format(self.__class__.__name__, info)


class Residue(object):

    def __init__(self, res_info):
        self.res_info = res_info
        #atoms_ = [Atom(i) for i in res_info]
        self.atom_dict = {}
        disorder = []
        for i in res_info:   # 排除disorder原子
            atom = Atom(i)
            if atom.name in self.atom_dict:
                continue
            else:
                self.atom_dict[atom.name] = atom
                disorder.append(atom.is_disorder)
            if atom.res_name == 'MSE':
                atom.res_name = 'MET'   # 把MSE残基改成MET
            
        if True in disorder:
            self.is_disorder = True
        else:
            self.is_disorder = False
            
        self.idx = self.atom_dict[atom.name].res_idx
        self.chain = self.atom_dict[atom.name].chain
        self.name = self.atom_dict[atom.name].res_name
        self.is_perfect = True if len(self.get_heavy_atoms)==len(RESIDUES_TOPO[self.name]) else False

    @property
    def to_heavy_string(self):
        return '\n'.join([a.to_string for a in self.get_heavy_atoms])
    
    @property
    def to_string(self):
        return '\n'.join([a.to_string for a in self.get_atoms])
    
    @property
    def get_coords(self):
        return np.array([a.coord for a in self.get_atoms])

    @property
    def get_atoms(self):
        return list(self.atom_dict.values())
    
    @property
    def get_heavy_atoms(self):
        return [a for a in self.atom_dict.values() if 'H' not in a.element]
    
    @property
    def get_heavy_coords(self):
        return np.array([a.coord for a in self.get_heavy_atoms])

    @property
    def center_of_mass(self):
        atom_mass = np.array([i.mass for i in self.get_atoms]).reshape(-1, 1)
        return np.sum(self.get_coords*atom_mass, axis=0)/atom_mass.sum()
    
    @property
    def bond_graph(self):
        i, j, bt = [], [], []
        res_graph = RESIDUES_TOPO[self.name]
        atom_names = [i.name for i in self.get_heavy_atoms]
        for ix, name in enumerate(atom_names):
            for adj in res_graph[name]:
                if adj in atom_names:
                    idx_j = atom_names.index(adj)
                    i.append(ix)
                    j.append(idx_j)
                    bt.append(res_graph[name][adj])
        edge_index = np.stack([i,j]).astype(dtype=np.int64)
        bt = np.array(bt, dtype=np.int64)
        return edge_index, bt
    
    @property
    def centroid(self):
        return self.get_coords.mean(axis=0)
    
    def __repr__(self):
        info = 'name={}, index={}, chain={}, is_disorder={}, is_perfect={}'.format(
            self.name, self.idx, self.chain, self.is_disorder, self.is_perfect
            )
        return '{}({})'.format(self.__class__.__name__, info)


class Chain(object):
    def __init__(self, chain_info, ignore_incomplete_res=True, pdb_file=None):
        self.pdb_file = pdb_file
        self.res_dict = {}
        '''if ignore_incomplete_res:
            self.residues = {}
            for i in chain_info:
                res = Residue(chain_info[i])
                if len(res.get_heavy_atoms) == len(RESIDUES_TOPO[res.name]):
                    self.residues[i] = res
        else:
            self.residues = {i:Residue(chain_info[i]) for i in chain_info}'''
        self.residues = {i:Residue(chain_info[i]) for i in chain_info}
        #print(self.residues)
        self.chain = list(self.residues.values())[0].chain
        self.__normalized__ = False
        self.center_of_mass_shift = None
        self.rotate_matrix = None
        self.ignore_incomplete_res =ignore_incomplete_res
    
    @property
    def get_incomplete_residues(self):
        return [i for i in self.residues.values() if i.is_perfect==False]

    @property
    def to_heavy_string(self):
        return '\n'.join([res.to_heavy_string for res in self.get_residues])
    
    @property
    def to_string(self):
        return '\n'.join([res.to_string for res in self.get_residues])
    
    @property
    def get_atoms(self):
        atoms = []
        for res in self.get_residues:
            atoms.extend(res.get_atoms)
        return atoms
    
    @property
    def get_residues(self):
        if self.ignore_incomplete_res:
            return [i for i in self.residues.values() if i.is_perfect]
        else:
            return list(self.residues.values())
        
    @property
    def get_heavy_atoms(self):
        atoms = []
        for res in self.get_residues:
            atoms.extend(res.get_heavy_atoms)
        return atoms
    
    @property
    def get_coords(self):
        return np.array([i.coord for i in self.get_atoms])

    @property
    def get_heavy_coords(self):
        return np.array([i.coord for i in self.get_heavy_atoms])

    @property
    def center_of_mass(self):
        atom_mass = np.array([i.mass for i in self.get_atoms]).reshape(-1, 1)
        return np.sum(self.get_coords*atom_mass, axis=0)/atom_mass.sum()

    @property
    def centroid(self):
        return self.get_coords.mean(axis=0)

    def compute_surface_atoms(self):
        path, filename = os.path.split(self.pdb_file)
        chain_file_name = filename.split('.')[0] + '_' + self.chain + '.pdb'
        chain_file = path+'/'+chain_file_name
        with open(chain_file, 'w') as fw:
            fw.write(self.to_heavy_string)

        pymol.cmd.load(chain_file)
        if path == '':
            path = './'
        sele = chain_file_name.split('.')[0]
        pymol.cmd.remove("({}) and hydro".format(sele))
        name = pymol.util.find_surface_atoms(sele=sele, _self=pymol.cmd)
        save_name = path + '/' + sele + '-surface.pdb'
        pymol.cmd.save(save_name,((name)))
        surf_protein = Protein(save_name, ignore_incomplete_res=False)
        surf_res_dict = {r.idx:r for r in surf_protein.get_residues}
        for res in self.get_residues:
            res_idx = res.idx
            if res_idx in surf_res_dict:
                for a in surf_res_dict[res_idx].get_heavy_atoms:
                    res.atom_dict[a.name].is_surf = True
        self.has_surf_atom = True
        os.remove(save_name)
        os.remove(chain_file)

    def get_surf_mask(self):
        if self.has_surf_atom is False:
            self.compute_surface_atoms()
        return np.array([a.is_surf for a in self.get_heavy_atoms], dtype=np.bool)

    def get_res_by_id(self, res_id):
        return self.residues[res_id]
    
    def __repr__(self):
        tmp = 'Chain={}, NumResidues={}, NumAtoms={}, NumHeavyAtoms={}'
        info = tmp.format(self.chain, len(self.residues), self.get_coords.shape[0], self.get_heavy_coords.shape[0])
        return '{}({})'.format(self.__class__.__name__, info)


class Protein(object):

    def __init__(self, pdb_file, ignore_incomplete_res=True):
        self.ignore_incomplete_res = ignore_incomplete_res
        self.name = pdb_file.split('/')[-1].split('.')[0]
        self.pdb_file = pdb_file
        self.has_surf_atom = False
        #self.pdb = [line.strip() for line in open(pdb_file).readlines()]
        #atoms = os.popen('grep ATOM {}'.format(pdb_file)).read().strip().split('\n')
        with open(pdb_file) as fr:
            lines = fr.readlines()
            surf_item = lines[0].strip().split()[-1]
            if surf_item in {'surf', 'inner'}:
                self.has_surf_atom = True
            chain_info = {}
            for line in lines:
                if line.startswith('ATOM'):
                    line = line.strip()
                    chain = line[21:22].strip()
                    res_idx = int(line[22:26].strip())
                    if chain not in chain_info:
                        chain_info[chain] = {}
                        chain_info[chain][res_idx] = [line]
                    elif res_idx not in chain_info[chain]:
                        chain_info[chain][res_idx] = [line]
                    else:
                        chain_info[chain][res_idx].append(line)
        
        self.chains = {
            c:Chain(chain_info[c], ignore_incomplete_res=ignore_incomplete_res, pdb_file=pdb_file) for c in chain_info
            }
        self.__normalized__ = False
        self.center_of_mass_shift = None
        self.rotate_matrix = None
    
    @property
    def get_incomplete_residues(self):
        res_list = []
        for i in self.chains:
            res_list += self.chains[i].get_incomplete_residues
        return res_list

    @property
    def to_heavy_string(self):
        return '\n'.join([res.to_heavy_string for res in self.get_residues])
    
    @property
    def to_string(self):
        return '\n'.join([res.to_string for res in self.get_residues])

    @property
    def get_residues(self,):
        res_list = []
        for i in self.chains:
            res_list += self.chains[i].get_residues
        return res_list

    @property
    def get_atoms(self):
        atoms = []
        for res in self.get_residues:
            atoms.extend(res.get_atoms)
        return atoms
    
    @property
    def get_heavy_atoms(self):
        atoms = []
        for res in self.get_residues:
            atoms.extend(res.get_heavy_atoms)
        return atoms

    @property
    def get_coords(self):
        return np.array([i.coord for i in self.get_atoms])

    @property
    def get_heavy_coords(self):
        return np.array([i.coord for i in self.get_heavy_atoms])

    @property
    def center_of_mass(self):
        atom_mass = np.array([i.mass for i in self.get_atoms]).reshape(-1, 1)
        return np.sum(self.get_coords*atom_mass, axis=0)/atom_mass.sum()
    
    @property
    def bond_graph(self):
        res_list = self.get_residues
        bond_index = []
        bond_type = []
        N_term_list = []
        C_term_list = []
        cusum = 0
        for ix, res in enumerate(res_list):
            e_idx, e_type = res.bond_graph
            bond_index.append(e_idx + cusum)
            bond_type.append(e_type)
            N_term_ix = [i.name for i in self.get_heavy_atoms].index('N') + cusum
            C_term_ix = [i.name for i in self.get_heavy_atoms].index('C') + cusum
            N_term_list.append(N_term_ix)
            C_term_list.append(C_term_ix)
            cusum += res.get_heavy_coords.shape[0]
            if ix != 0:
                if res.idx-res_list[ix-1].idx == 1 and res.chain == res_list[ix-1].chain:
                    bond_idx_between_res = np.array(
                        [[N_term_ix, C_term_list[ix-1]],[C_term_list[ix-1],N_term_ix]],
                        dtype=np.long
                    )
                    bond_index.append(bond_idx_between_res)
                    bond_type_between_res = np.array([1,1], dtype=np.long)
                    bond_type.append(bond_type_between_res)
        bond_index = np.concatenate(bond_index, axis=1)
        bond_type = np.concatenate(bond_type)
        return bond_index, bond_type

    @property
    def centroid(self):
        return self.get_coords.mean(axis=0)
    
    def get_chain(self, chain_id):
        return self.chains[chain_id]
    
    def get_res_by_id(self, chain_id, res_id):
        return self.chains[chain_id].get_res_by_id(res_id)
    
    def get_atom_dict(self, removeHs=True, get_surf=False):
        atom_dict = {
            'element': [],
            'pos': [],
            'is_backbone': [],
            'atom_name': [],
            'atom_to_aa_type': []
             }
        for a in self.get_atoms:
            if a.element == 'H' and removeHs:
                continue
            atom_dict['element'].append(a.to_dict['element'])
            atom_dict['pos'].append(a.to_dict['pos'])
            atom_dict['is_backbone'].append(a.to_dict['is_backbone'])
            atom_dict['atom_name'].append(a.to_dict['atom_name'])
            atom_dict['atom_to_aa_type'].append(a.to_dict['atom_to_aa_type'])
        atom_dict['element'] = np.array(atom_dict['element'], dtype=np.long)
        atom_dict['pos'] = np.array(atom_dict['pos'], dtype=np.float32)
        atom_dict['is_backbone'] = np.array(atom_dict['is_backbone'], dtype=np.bool)
        #atom_dict['atom_name'] = atom_dict['atom_name']
        if get_surf:
            atom_dict['surface_mask'] = np.array([a.is_surf for a in self.get_heavy_atoms], dtype=np.bool)#self.get_surf_mask()
        
        atom_dict['atom_to_aa_type'] = np.array(atom_dict['atom_to_aa_type'], dtype=np.long)
        atom_dict['molecule_name'] = None
        protein_bond_index, protein_bond_type = self.bond_graph
        atom_dict['bond_index'] = protein_bond_index
        atom_dict['bond_type'] = protein_bond_type
        atom_dict['filename'] = self.pdb_file
        return atom_dict

    def get_backbone_dict(self, removeHs=True):
        atom_dict = self.get_atom_dict(removeHs=removeHs)
        backbone_dict = {}
        backbone_dict['element'] = atom_dict['element'][atom_dict['is_backbone']]
        backbone_dict['pos'] = atom_dict['pos'][atom_dict['is_backbone']]
        backbone_dict['is_backbone'] = np.ones(atom_dict['is_backbone'].sum(), dtype=np.bool)
        backbone_dict['atom_name'] = np.array(atom_dict['atom_name'])[atom_dict['is_backbone']].tolist()
        backbone_dict['atom_to_aa_type'] = atom_dict['atom_to_aa_type'][atom_dict['is_backbone']]
        backbone_dict['molecule_name'] = atom_dict['molecule_name']
        atom_dict['bond_index'] = np.empty([0,2], dtype=np.long)
        atom_dict['bond_type'] = np.empty(0, dtype=np.long)
        atom_dict['filename'] = self.pdb_file
        return backbone_dict

    @property
    def get_backbone(self):
        atoms = []
        for res in self.get_residues:
            bkb = [a for a in res.get_atoms if a.name in BACKBONE_SYMBOL]
            atoms += bkb
        return atoms

    def compute_surface_atoms(self):
        """
        If the pdb_file is a pocket_file, the surface atoms is not correct!
        """
        pymol.cmd.load(self.pdb_file)
        path, filename = os.path.split(self.pdb_file)
        if path == '':
            path = './'
        sele = filename.split('.')[0]
        pymol.cmd.remove("({}) and hydro".format(sele))
        name = pymol.util.find_surface_atoms(sele=sele, _self=pymol.cmd)
        save_name = path + '/' + sele + '-surface.pdb'
        pymol.cmd.save(save_name,((name)))
        surf_protein = Protein(save_name, ignore_incomplete_res=False)
        surf_res_dict = {r.idx:r for r in surf_protein.get_residues}
        for res in self.get_residues:
            res_idx = res.idx
            if res_idx in surf_res_dict:
                for a in surf_res_dict[res_idx].get_heavy_atoms:
                    res.atom_dict[a.name].is_surf = True
        self.has_surf_atom = True
        os.remove(save_name)

    def get_surf_mask(self):
        if self.has_surf_atom is False:
            self.compute_surface_atoms()
        return np.array([a.is_surf for a in self.get_heavy_atoms], dtype=np.bool)
        
    @staticmethod
    def empty_dict():
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
        return empty_pocket_dict
    
    def __repr__(self):
        num_res = 0
        num_atom = 0
        for i in self.chains:
            res_list = list(self.chains[i].residues.values())
            num_res += len(res_list)
            for i in res_list:
                num_atom += len(i.get_heavy_atoms)
        num_incomp = len(self.get_incomplete_residues)
        tmp = 'Name={}, NumChains={}, NumResidues={}, NumHeavyAtoms={}, NumIncompleteRes={}'
        info = tmp.format(
            self.name, len(self.chains), num_res, num_atom, num_incomp
            )
        return '{}({})'.format(self.__class__.__name__, info)


####################################################################################################################


ATOM_FAMILIES = ['Acceptor', 'Donor', 'Aromatic', 'Hydrophobe', 'LumpedHydrophobe', 'NegIonizable', 'PosIonizable', 'ZnBinder']
ATOM_FAMILIES_ID = {s: i for i, s in enumerate(ATOM_FAMILIES)}
BOND_TYPES = {t: i for i, t in enumerate(BondType.names.values())}
BOND_NAMES = {i: t for i, t in enumerate(BondType.names.keys())}


def is_in_ring(mol):
    d = {a:np.array([], dtype=np.int64) for a in range(len(mol.GetAtoms()))}
    rings = Chem.GetSymmSSSR(mol)
    for a in d:
        for r_idx, ring in enumerate(rings):
            if a in ring:
                d[a] = np.append(d[a], r_idx+1)
            else:
                d[a] = np.append(d[a], -a)
    return d

def parse_sdf_to_dict(mol_file):
    fdefName = os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
    rdmol = next(iter(Chem.SDMolSupplier(mol_file, removeHs=True)))
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
    
    return {'element': element,
            'pos': pos,
            'bond_index': edge_index,
            'bond_type': edge_type,
            'center_of_mass': center_of_mass,
            'atom_feature': feat_mat,
            'ring_info': ring_info,
            'filename':mol_file
           }


from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform


class Ligand(object):
    def __init__(self, mol_file, removeHs=True, sanitize=True):
        if isinstance(mol_file, Chem.rdchem.Mol):
            mol = mol_file
            self.name = None
            self.lig_file = None
        else:
            mol = Chem.MolFromMolFile(mol_file, removeHs=removeHs, sanitize=sanitize)
            if mol is None:
                mol = Chem.MolFromMolFile(mol_file, removeHs=removeHs, sanitize=False)
                mol.UpdatePropertyCache(strict=False)
                Chem.SanitizeMol(
                    mol,  ## if raise error, we can try Chem.rdmolops.SanitizeFlags.SANITIZE_FINDRADICALS
                    Chem.SanitizeFlags.SANITIZE_FINDRADICALS|\
                    Chem.SanitizeFlags.SANITIZE_KEKULIZE|\
                    Chem.SanitizeFlags.SANITIZE_SETAROMATICITY| \
                    Chem.SanitizeFlags.SANITIZE_SETCONJUGATION| \
                    Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION| \
                    Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
                    catchErrors=True
                    )
            self.name = mol_file.split('/')[-1].split('.')[0]
            self.lig_file = mol_file
            
        Chem.Kekulize(mol)
        self.mol = mol
        #self.self_loop = self_loop
        self.num_atoms = len(self.mol.GetAtoms())
        self.normalized_coords = None
    
    def normalize_pos(self, shift_vector, rotate_matrix):
        conformer = self.mol.GetConformer()
        coords = np.array([conformer.GetAtomPosition(a.GetIdx()) for a in self.mol.GetAtoms()])
        coords = (coords - shift_vector)@rotate_matrix
        for ix, pos in enumerate(coords):
            conformer.SetAtomPosition(ix, pos)
        self.normalized_coords = coords
    
    def mol_block(self):
        return Chem.MolToMolBlock(self.mol)
    
    def to_dict(self):
        fdefName = os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef')
        factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
        #rdmol = next(iter(Chem.SDMolSupplier(mol_file, removeHs=True)))
        ring_info = is_in_ring(self.mol)
        conformer = self.mol.GetConformer()
        feat_mat = np.zeros([self.mol.GetNumAtoms(), len(ATOM_FAMILIES)], dtype=np.int64)
        for feat in factory.GetFeaturesForMol(self.mol):
            feat_mat[feat.GetAtomIds(), ATOM_FAMILIES_ID[feat.GetFamily()]] = 1
        
        element, pos, atom_mass = [], [], []
        for a in self.mol.GetAtoms():
            element.append(a.GetAtomicNum())
            pos.append(conformer.GetAtomPosition(a.GetIdx()))
            atom_mass.append(a.GetMass())
        element = np.array(element, dtype=np.int64)
        pos = np.array(pos, dtype=np.float32)
        atom_mass = np.array(atom_mass, np.float32)
        center_of_mass = (pos * atom_mass.reshape(-1,1)).sum(0)/atom_mass.sum()
        
        edge_index, edge_type = [], []
        for b in self.mol.GetBonds():
            row = [b.GetBeginAtomIdx(), b.GetEndAtomIdx()]
            col = [b.GetEndAtomIdx(), b.GetBeginAtomIdx()]
            edge_index.extend([row, col])
            edge_type.extend([BOND_TYPES[b.GetBondType()]] * 2)
        edge_index = np.array(edge_index)
        edge_index_perm = edge_index[:,0].argsort()
        edge_index = edge_index[edge_index_perm].T
        edge_type = np.array(edge_type)[edge_index_perm]
        
        return {'element': element,
                'pos': pos,
                'bond_index': edge_index,
                'bond_type': edge_type,
                'center_of_mass': center_of_mass,
                'atom_feature': feat_mat,
                'ring_info': ring_info,
                'filename':self.lig_file
            }

    @staticmethod
    def empty_dict():
        empty_ligand_dict = {}
        empty_ligand_dict = Dict(empty_ligand_dict)
        empty_ligand_dict.element = np.empty(0, dtype=np.int64)
        empty_ligand_dict.pos = np.empty([0,3], dtype=np.float32)
        empty_ligand_dict.bond_index = np.empty([2,0], dtype=np.int64)
        empty_ligand_dict.bond_type = np.empty(0, dtype=np.int64)
        empty_ligand_dict.center_of_mass = np.empty([0,3], dtype=np.float32)
        empty_ligand_dict.atom_feature = np.empty([0,8], dtype=np.float32)
        empty_ligand_dict.ring_info = {}
        empty_ligand_dict.filename = None
        return empty_ligand_dict

    def __repr__(self):
        tmp = 'Name={}, NumAtoms={}'
        info = tmp.format(self.name, self.num_atoms)
        return '{}({})'.format(self.__class__.__name__, info)