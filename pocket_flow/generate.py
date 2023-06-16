import torch
from torch_geometric.nn import knn, radius
import time
from rdkit import Chem
from .utils import get_tri_edges
from .utils import add_ligand_atom_to_data, data2mol, modify, check_valency, check_alert_structures
from .gdbp_model import embed_compose
from .utils import verify_dir_exists, substructure
from rdkit import RDLogger


RDLogger.DisableLog('rdApp.*')
max_valence_dict = {1:1, 6:4, 7:3, 8:2, 9:1, 15:5, 16:6, 17:1, 35:1, 53:1}

class Generate(object):
    def __init__(self, model, transform, temperature=[1.0, 1.0], atom_type_map=[6,7,8,9,15,16,17,35,53],
                 num_bond_type=4, max_atom_num=35, focus_threshold=0.5, max_double_in_6ring=0, min_dist_inter_mol=3.0, bond_length_range=(1.0, 2.0),
                 choose_max=True, device='cuda:0'):
       
        self.model = model
        self.transform = transform
        self.temperature = temperature
        self.atom_type_map = atom_type_map
        self.num_bond_type = num_bond_type
        self.max_atom_num = max_atom_num
        self.focus_threshold = focus_threshold
        self.max_double_in_6ring = max_double_in_6ring
        self.min_dist_inter_mol = min_dist_inter_mol
        self.bond_length_range = bond_length_range
        self.choose_max = choose_max
        self.hidden_channels = model.config.hidden_channels
        self.knn = model.config.encoder.knn
        self.device = device
        self.bond_type_map =  {
            1: Chem.rdchem.BondType.SINGLE, 
            2: Chem.rdchem.BondType.DOUBLE, 
            3: Chem.rdchem.BondType.TRIPLE
            }

    @staticmethod
    def __choose_focal(focal_net, h_ctx, ctx_idx, focus_threshold, choose_max, surf_mask=None):
        focal_pred = focal_net(h_ctx, ctx_idx)
        focal_prob = torch.sigmoid(focal_pred).view(-1)
        if choose_max:
            max_idx = focal_pred.argmax()
            focal_idx_candidate = ctx_idx[max_idx].view(-1)
            focal_prob = focal_prob[max_idx].view(-1)
        else:
            if isinstance(surf_mask, torch.Tensor) and surf_mask.sum() > 0:
                surf_idx = torch.nonzero(surf_mask).view(-1)
                focal_prob_surf = focal_prob[surf_mask]
                surf_focal_mask = focal_prob_surf > focus_threshold
                focal_idx_candidate = surf_idx[surf_focal_mask]
                focal_prob = focal_prob_surf[surf_focal_mask]
                if surf_focal_mask.sum() == 0:
                    return False, focal_prob
            else:
                focal_mask = (focal_prob >= focus_threshold).view(-1)
                focal_idx_candidate = ctx_idx[focal_mask]
                focal_prob = focal_prob[focal_mask]
                if focal_mask.sum() == 0:
                    return False, focal_prob
        return focal_idx_candidate, focal_prob

    def choose_focal(self, h_cpx, cpx_index, idx_ligand_ctx_in_cpx, data, atom_idx):
        if atom_idx == 0:
            if 'protein_surface_mask' in data:
                surf_mask = data.protein_surface_mask
            else:
                surf_mask = None
            focal_idx_, focal_prob = self.__choose_focal(
                self.model.focal_net, h_cpx, cpx_index, 
                self.focus_threshold, self.choose_max, #1
                surf_mask=surf_mask)
        else:
            focal_idx_, focal_prob = self.__choose_focal(
                self.model.focal_net, h_cpx, idx_ligand_ctx_in_cpx, 
                self.focus_threshold, 1#self.choose_max
                )
        if focal_idx_ is False:
            return False
        
        # valence_check for focal atom
        focal_valence_check = False
        if data.ligand_context_element.size(0) > 3 and focal_idx_ is not False and atom_idx != 0:
            max_valence = data.max_atom_valence[focal_idx_]
            valence_in_ligand_context_focal = data.ligand_context_valence[focal_idx_]
            valence_mask = max_valence > valence_in_ligand_context_focal
            focal_idx_ = focal_idx_[valence_mask]
            focal_prob = focal_prob[valence_mask]
            focal_valence_check = valence_mask.sum() == 0
            if focal_valence_check:
                return False
            else:
                self.counter += 1
        if focal_valence_check:
            return False
        return focal_idx_, focal_prob

    def atom_generate(self, h_cpx, focal_idx, focal_prob, atom_idx):
        latent_atom = self.prior_node.sample([focal_idx.size(0)])
        latent_atom = self.model.atom_flow.reverse(latent_atom, h_cpx, focal_idx)
        if self.choose_max:
            if atom_idx == 0:
                new_atom_type_prob, new_atom_type_pool = torch.max(latent_atom, -1)
                new_atom_idx_with_max_prob = torch.flip(new_atom_type_prob.argsort(), dims=[0])#[:10]
                new_atom_type_pool = new_atom_type_pool[new_atom_idx_with_max_prob]
                new_atom_type_prob = torch.log(new_atom_type_prob[new_atom_idx_with_max_prob])

                focal_idx_ = focal_idx[new_atom_idx_with_max_prob]
                focal_choose_idx = torch.multinomial(focal_prob, 1)
                focal_idx_ = focal_idx[focal_choose_idx]
                new_atom_type = new_atom_type_pool[focal_choose_idx]
            else:
                new_atom_type_prob, new_atom_type = torch.max(latent_atom, -1)
                focal_idx_ = focal_idx
        else:
            new_atom_type_prob, new_atom_type_pool = torch.max(latent_atom, -1)
            new_atom_idx_with_max_prob = torch.flip(new_atom_type_prob.argsort(), dims=[0])#[:10]
            new_atom_type_pool = new_atom_type_pool[new_atom_idx_with_max_prob]
            new_atom_type_prob = torch.log(new_atom_type_prob[new_atom_idx_with_max_prob])

            focal_idx_ = focal_idx[new_atom_idx_with_max_prob]
            focal_choose_idx = torch.multinomial(focal_prob, 1)
            focal_idx_ = focal_idx[focal_choose_idx]
            new_atom_type = new_atom_type_pool[focal_choose_idx]
        return new_atom_type, focal_idx_
    
    def pos_generate(self, h_cpx, atom_type_emb, focal_idx, cpx_pos, atom_idx):
        new_relative_pos, new_abs_pos, sigma, pi = self.model.pos_predictor(
                        h_cpx, 
                        focal_idx,
                        cpx_pos, 
                        atom_type_emb=atom_type_emb
                        )
        new_relative_pos = new_relative_pos.view(-1,3)
        new_abs_pos = new_abs_pos.view(-1,3)
        pi = pi.view(-1)
        dist = torch.norm(new_relative_pos, p=2, dim=-1)

        if atom_idx != 0:
            dist_mask = (dist>self.bond_length_range[0]) == (dist<self.bond_length_range[1])
            new_pos_to_add_ = new_abs_pos[dist_mask]
            check_pos = new_pos_to_add_.size(0) != 0
            if check_pos:
                pi_ = pi[dist_mask]
                pos_choose_idx = torch.multinomial(pi_, 1)
                new_pos_to_add = new_pos_to_add_[pos_choose_idx]
                return new_pos_to_add
            else:
                return False
        else:
            dist_mask = dist > self.min_dist_inter_mol # inter-molecular dist cutoff
            new_pos_to_add_ = new_abs_pos[dist_mask]
            check_pos = new_pos_to_add_.size(0) != 0
            if check_pos:
                pi_ = pi[dist_mask]
                pos_choose_idx = torch.multinomial(pi_, 1)
                new_pos_to_add = new_pos_to_add_[pos_choose_idx]
                return new_pos_to_add
            else:
                return False

    def bond_generate(self, h_cpx, data, new_pos_to_add, atom_type_emb, atom_idx, rw_mol):
        if atom_idx == 0:
            is_check = False
            new_edge_idx = torch.empty([2, 0], dtype=torch.long)
            new_bond_type_to_add = torch.empty([0], dtype=torch.long)
        else:
            edge_index_query = radius(data.ligand_context_pos, new_pos_to_add, r=4.0, num_workers=16)
            pos_query_knn_edge_idx = knn(
                x=data.cpx_pos, y=new_pos_to_add, k=self.model.config.encoder.knn, num_workers=16
                )
            index_real_cps_edge_for_atten, tri_edge_index, tri_edge_feat = get_tri_edges(
                edge_index_query, 
                new_pos_to_add, 
                data.context_idx, 
                data.ligand_context_bond_index, 
                data.ligand_context_bond_type
                )
            resample_edge = 0
            no_bond = True
            while no_bond:
                if resample_edge >= 50:
                    self.resample_edge_faild = True
                    rw_mol.RemoveAtom(atom_idx)
                    return False
                latent_edge = self.prior_edge.sample([edge_index_query.size(1)])
                edge_latent = self.model.edge_flow.reverse(
                    edge_latent=latent_edge,
                    pos_query=new_pos_to_add, 
                    edge_index_query=edge_index_query, 
                    cpx_pos=data.cpx_pos, 
                    node_attr_compose=h_cpx, 
                    edge_index_q_cps_knn=pos_query_knn_edge_idx, 
                    index_real_cps_edge_for_atten=index_real_cps_edge_for_atten, 
                    tri_edge_index=tri_edge_index, 
                    tri_edge_feat=tri_edge_feat,
                    atom_type_emb=atom_type_emb
                    )
                edge_pred_type = edge_latent.argmax(-1)
                edge_pred_mask = edge_pred_type > 0
                if edge_pred_mask.sum() > 0:
                    new_bond_type_to_add = edge_pred_type[edge_pred_mask]
                    new_edge_idx = edge_index_query[:,edge_pred_mask]

                    new_edge_vec = new_pos_to_add[new_edge_idx[0]]-data.cpx_pos[new_edge_idx[1]]
                    new_edge_dist = torch.norm(new_edge_vec, p=2, dim=-1)
                    if (new_edge_dist > self.bond_length_range[1]).sum() > 0:
                        if resample_edge >= 50:
                            self.resample_edge_faild = True
                            rw_mol.RemoveAtom(atom_idx)
                            return False
                        else:
                            resample_edge += 1
                            continue
                        
                    for ix in range(new_edge_idx.size(1)):
                        i, j = new_edge_idx[:, ix].tolist()
                        bond_type = new_bond_type_to_add[ix].item()
                        rw_mol.AddBond(atom_idx, j, self.bond_type_map[bond_type])
                    valency_valid = check_valency(rw_mol)
                    if valency_valid:
                        no_bond = False
                        break
                    else:
                        for ix in range(new_edge_idx.size(1)):
                            i, j = new_edge_idx[:, ix].tolist()
                            bond_type = new_bond_type_to_add[ix].item()
                            rw_mol.RemoveBond(atom_idx, j)
                        resample_edge += 1
                        #print(resample_edge)
                        if resample_edge >= 50:
                            self.resample_edge_faild = True
                            rw_mol.RemoveAtom(atom_idx)
                            return False
                else:
                    resample_edge += 1
        return rw_mol, new_edge_idx, new_bond_type_to_add

    def run(self, data):
        data.max_atom_valence = torch.empty(0, dtype=torch.long)
        data = data.to(self.device)
        with torch.no_grad():
            self.prior_node = torch.distributions.normal.Normal(
                    torch.zeros([len(self.atom_type_map)]).cuda(data.cpx_pos.device), 
                    self.temperature[0] * torch.ones([len(self.atom_type_map)]).cuda(data.cpx_pos.device)
                    )
            self.prior_edge = torch.distributions.normal.Normal(
                    torch.zeros([self.num_bond_type]).cuda(data.cpx_pos.device), 
                    self.temperature[1] * torch.ones([self.num_bond_type]).cuda(data.cpx_pos.device)
                    )
            
            rw_mol = Chem.RWMol()
            self.counter = 0
            for atom_idx in range(self.max_atom_num):
                data = data.to(self.device)
                h_cpx = embed_compose(
                    data.cpx_feature.float(), data.cpx_pos, data.idx_ligand_ctx_in_cpx, 
                    data.idx_protein_in_cpx, self.model.ligand_atom_emb, 
                    self.model.protein_atom_emb, self.model.emb_dim
                    )
                # encoding context
                h_cpx = self.model.encoder(
                    node_attr = h_cpx,
                    pos = data.cpx_pos,
                    edge_index = data.cpx_edge_index,
                    edge_feature = data.cpx_edge_feature
                )

                self.resample_edge_faild = False
                self.check_node = True
                self.resample_node = 0
                while self.check_node:
                    if self.resample_node > 50:
                        break
                    
                    # choose focal
                    focal_out = self.choose_focal(
                            h_cpx, data.idx_protein_in_cpx, data.idx_ligand_ctx_in_cpx, 
                            data, atom_idx   # cpx_backbone_index
                            )
                    if focal_out is False:
                        break
                    else:
                        focal_idx, focal_prob = focal_out
                    
                    # generate atom
                    new_atom_type, focal_idx = self.atom_generate(
                        h_cpx, focal_idx, focal_prob, atom_idx
                        )
                    # get position of new atom
                    atom_type_emb = self.model.atom_type_embedding(new_atom_type).view(-1, self.hidden_channels)
                    new_pos_to_add = self.pos_generate(
                        h_cpx, atom_type_emb, focal_idx, data.cpx_pos, atom_idx
                        )
                    if new_pos_to_add is False:
                        self.resample_node += 1
                        continue
                    else:
                        rw_mol.AddAtom(Chem.Atom(self.atom_type_map[new_atom_type]))

                    # generate bonds
                    bond_out = self.bond_generate(
                        h_cpx, data, new_pos_to_add, atom_type_emb, atom_idx, rw_mol
                        )
                    if bond_out is not False:
                        rw_mol, new_edge_idx, new_bond_type_to_add = bond_out
                        #has_alert = check_alert_structures(rw_mol, ['[O]-[O]','[N]-[O]'])
                        has_alert = check_alert_structures(rw_mol, ['[O]-[O]','[N]-[O,Br,Cl,I,F,P]','[S,P]-[Br,Cl,I,F]','[P]-[O]-[P]','[Br,Cl,I,F]-[Br,Cl,I,F]'])
                        if has_alert:
                            for ix in range(new_edge_idx.size(1)):
                                i, j = new_edge_idx[:, ix].tolist()
                                rw_mol.RemoveBond(atom_idx, j)
                            rw_mol.RemoveAtom(atom_idx)
                            self.resample_node += 1
                            continue
                        else:
                            break
                    if self.resample_edge_faild:
                        break
                
                if self.resample_edge_faild or self.resample_node > 50 or focal_out is False:
                    break
                else:
                    data = data.to('cpu')
                    data = add_ligand_atom_to_data(
                        data, 
                        new_pos_to_add.to('cpu'), 
                        new_atom_type.to('cpu'), 
                        new_edge_idx.to('cpu'), 
                        new_bond_type_to_add.to('cpu'), 
                        type_map=self.atom_type_map
                        )
                    data = self.transform(data)
        #mol_ = rw_mol.GetMol()
        try:
            mol = data2mol(data)
            modified_mol = modify(mol, max_double_in_6ring=self.max_double_in_6ring)
            return modified_mol, mol
        except:
            mol_ = rw_mol.GetMol()
            print('Invalid mol: ', Chem.MolToSmiles(mol_))
            mol = data2mol(data)
            return None

    def generate(self, data, num_gen=100, rec_name='recptor', with_print=True, root_path='gen_results'):
        date = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        out_dir = root_path+'/'+rec_name+'/'+date+'/'
        self.out_dir = out_dir
        verify_dir_exists(out_dir)
        valid_mol = []
        smiles_list = []
        valid_conuter = 0
        for i in range(num_gen):
            data_clone = data.clone().detach()
            out = self.run(data_clone)
            if out:
                mol, mol_NoModify = out
            del data_clone
            if mol is not None:
                #print(len(mol.GetAtoms()))
                mol.SetProp('_Name', 'No_%s-%s'%(valid_conuter, out_dir))
                mol_NoModify.SetProp('_Name', 'No_%s-%s' %(valid_conuter, out_dir))
                smi = Chem.MolToSmiles(mol)
                smi_NoModify = Chem.MolToSmiles(mol_NoModify)
                if with_print:
                    print(smi)
                with open(out_dir+'generated.sdf', 'a') as sdf_writer:
                    mol_block = Chem.MolToMolBlock(mol)
                    sdf_writer.write(mol_block + '\n$$$$\n')
                with open(out_dir+'generated_NoModify.sdf', 'a') as sdf_writer1:
                    mol_block_NoModify = Chem.MolToMolBlock(mol_NoModify)
                    sdf_writer1.write(mol_block_NoModify + '\n$$$$\n')
                with open(out_dir+'generated.smi','a') as smi_writer:
                    smi_writer.write(smi+'\n')
                with open(out_dir+'generated_NoModify.smi','a') as smi_writer1:
                    smi_writer1.write(smi_NoModify+'\n')
                smiles_list.append(smi)
                valid_mol.append(mol)
                valid_conuter += 1
        print(len(smiles_list))
        print(len(set(smiles_list)))
        print('Validity: {:.4f}'.format(len(smiles_list)/num_gen))
        print('Unique: {:.4f}'.format(len(set(smiles_list))/len(smiles_list)))
        out_statistic = {
            'Validity':len(smiles_list)/num_gen,
            'Unique':len(set(smiles_list))/len(smiles_list)
            }
        ring_size_statis = substructure([valid_mol])
        out_statistic['ring_size'] = ring_size_statis
        with open(out_dir+'metrics.dir','w') as fw:
            fw.write(str(out_statistic))
