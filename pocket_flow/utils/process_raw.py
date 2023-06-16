import os 
import glob
import numpy as np
from multiprocessing import Pool
from .ParseFile import Protein, parse_sdf_to_dict, Ligand
from .residues_base import RESIDUES_TOPO


def verify_dir_exists(dirname):
    if os.path.isdir(os.path.dirname(dirname)) == False:
        os.makedirs(os.path.dirname(dirname))

def ComputeDistMat(m1, m2):
    #m1_square = np.sum(m1*m1, axis=1, keepdims=True)
    m1_square = np.expand_dims(np.einsum('ij,ij->i', m1, m1), axis=1)
    if m1 is m2:
        m2_square = m1_square.T
    else:
        #m2_square = np.sum(m2*m2, axis=1, keepdims=True).T
        m2_square = np.expand_dims(np.einsum('ij,ij->i', m2, m2), axis=0)
    dist_mat = m1_square + m2_square - np.dot(m1, m2.T)*2
    # result maybe less than 0 due to floating point rounding errors.
    dist_mat = np.maximum(dist_mat, 0, dist_mat)
    if m1 is m2:
        # Ensure that distances between vectors and themselves are set to 0.0.
        # This may not be the case due to floating point rounding errors.
        dist_mat.flat[::dist_mat.shape[0] + 1] = 0.0
    dist_mat = np.sqrt(dist_mat)
    return dist_mat

class SplitPocket(object):
    
    def __init__(self, 
                 main_path='./data/CrossDocked2020/',
                 sample_path='/Samples/',
                 new_sapmle_path='/Samples_Pocket/',
                 type_file='./data/CrossDocked2020/samples.types',
                 dist_cutoff=10,
                 get_surface_atom=True):
        
        self.main_path = main_path
        self.sample_path = sample_path
        self.new_sapmle_path = new_sapmle_path
        self.dist_cutoff = dist_cutoff
        self.type_file = type_file
        self.types = [line.strip().split() for line in open(type_file).readlines()]
        self.exceptions = []
        self.get_surface_atom = get_surface_atom

    @staticmethod  
    def _split_pocket(protein, ligand, dist_cutoff):
        res = np.array(protein.get_residues)
        cm_res = np.array([r.center_of_mass for r in res])
        #dist_mat = ComputeDistMat(ligand.normalized_coords, cm_res)
        lig_conformer = ligand.mol.GetConformer()
        lig_pos = [lig_conformer.GetAtomPosition(a.GetIdx()) for a in ligand.mol.GetAtoms()]
        dist_mat = ComputeDistMat(lig_pos, cm_res)
        bool_dist_mat = dist_mat < dist_cutoff
        pocket_res = res[bool_dist_mat.sum(axis=0)>0]
        pocket_block = '\n'.join(
            [i.to_heavy_string for i in pocket_res if len(i.get_heavy_atoms)==len(RESIDUES_TOPO[i.name])]
            )
        return pocket_block, ligand.mol_block()

    @staticmethod  
    def _split_pocket_with_surface_atoms(protein, ligand, dist_cutoff):

        protein.compute_surface_atoms()

        res = np.array(protein.get_residues)
        cm_res = np.array([r.center_of_mass for r in res])

        lig_conformer = ligand.mol.GetConformer()
        lig_pos = [lig_conformer.GetAtomPosition(a.GetIdx()) for a in ligand.mol.GetAtoms()]
        dist_mat = ComputeDistMat(lig_pos, cm_res)
        bool_dist_mat = dist_mat < dist_cutoff
        pocket_res = res[bool_dist_mat.sum(axis=0)>0]
        pocket_block = []
        for r in pocket_res:
            for a in r.get_heavy_atoms:
                if a.is_surf is True:
                    pocket_block.append(a.to_string + ' surf')
                else:
                    pocket_block.append(a.to_string + ' inner')
        pocket_block = '\n'.join(pocket_block)
        return pocket_block, ligand.mol_block()

    def _do_split(self, items):
        try:
            sub_path = items[4].split('/')[0]
            ligand_name = items[4].split('/')[1].split('.')[0]
            chain_id = items[3].split('.')[0].split('_')[-2]

            protein = Protein(self.main_path+self.sample_path+items[3])
            chain = protein.get_chain(chain_id)
            ligand = Ligand(self.main_path+self.sample_path+items[4], sanitize=True)
            
            if self.get_surface_atom:
                pocket_block, ligand_block = self._split_pocket_with_surface_atoms(chain, ligand, self.dist_cutoff)
            else:
                pocket_block, ligand_block = self._split_pocket(chain, ligand, self.dist_cutoff)

            save_path = '{}/{}/{}/'.format(self.main_path, self.new_sapmle_path, sub_path)
            verify_dir_exists(save_path)
            pocket_file_name = '{}/{}_pocket{}.pdb'.format(save_path, ligand_name, self.dist_cutoff)

            open(pocket_file_name, 'w').write(pocket_block)
            open(save_path+'/{}.mol'.format(ligand_name), 'w').write(ligand_block)
        except:
            protein_file = self.main_path+self.sample_path+items[3]
            ligand_file = self.main_path+self.sample_path+items[4]
            print('[Exception]', protein_file, ligand_file)

    @staticmethod
    def split_pocket_from_site_map(site_map, protein_file, dist_cutoff):
        '''
        if target dosn't have ligand, we can use the site_map.pdb computed by schrodinger or other software
        to split potential pocket.
        '''
        site_coords = []
        with open(site_map) as fr:
            lines = fr.readlines()
            for line in lines:
                if line.startswith('HETATM'):
                    line = line.strip()
                    xyz = [float(line[30:38].strip()),
                            float(line[38:46].strip()),
                            float(line[46:54].strip())]
                    site_coords.append(xyz)
        site_coords = np.array(site_coords)
        
        protein = Protein(protein_file)
        res = np.array(protein.get_residues)
        cm_res = np.array([r.center_of_mass for r in res])
        dist_mat = ComputeDistMat(site_coords, cm_res)
        bool_dist_mat = dist_mat < dist_cutoff
        pocket_res = res[bool_dist_mat.sum(axis=0)>0]
        pocket_block = '\n'.join(
            [i.to_heavy_string for i in pocket_res if len(i.get_heavy_atoms)==len(RESIDUES_TOPO[i.name])]
            )
        return pocket_block

    def __call__(self, np=10):
        pool = Pool(processes=np)    
        data_pool = pool.map(self._do_split, self.types)
        pool.close()
        pool.join()
        print('Done !')