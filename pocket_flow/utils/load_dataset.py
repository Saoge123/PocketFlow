import os
import pickle
import glob
import lmdb
import random
from rdkit import Chem
from multiprocessing import Pool
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
#from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm
from .data import ComplexData, torchify_dict
from .ParseFile import Protein, Ligand, parse_sdf_to_dict
 


class LoadDataset(Dataset):

    def __init__(self, dataset_path, transform=None, map_size=10*(1024*1024*1024)):
        super().__init__()
        self.dataset_path = dataset_path
        self.transform = transform
        self.map_size = map_size
        self.db = None
        self.keys = None

    def _connect_db(self):
        self.db = lmdb.open(
            self.dataset_path,
            map_size=self.map_size,   # 10GB
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.db.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))
    
    def _close_db(self):
        self.db.close()
        self.db = None
        self.keys = None
    
    def __len__(self):
        if self.db is None:
            self._connect_db()
        return len(self.keys)

    def __getitem__(self, idx):
        if self.db is None:
            self._connect_db()
        key = self.keys[idx]
        data = pickle.loads(self.db.begin().get(key))
        #data.id = idx
        #assert data.protein_pos.size(0) > 0
        if self.transform is not None:
            data = self.transform(data)
        return data
    
    def __connect_db_edit__(self):
        self.db = lmdb.open(
            self.dataset_path,
            map_size=self.map_size,   # 10GB
            create=False,
            subdir=False,
            readonly=False,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.db.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))
    
    def remove(self, idx):
        if self.db is None:
            self.__connect_db_edit__()
        txn = self.db.begin(write=True)
        txn.delete(self.keys[idx])
        txn.commit()
        self._close_db()

    @staticmethod
    def split(dataset, val_num=None, shuffle=True, random_seed=0):
        index = list(range(len(dataset)))
        if shuffle:
            random.seed(random_seed)
            random.shuffle(index)
        index = torch.LongTensor(index)
        split_dic = {'valid':index[:val_num], 'train':index[val_num:]}
        subsets = {k:Subset(dataset, indices=v) for k, v in split_dic.items()}
        train_set, val_set = subsets['train'], subsets['valid']
        return train_set, val_set
    
    @staticmethod
    def split_by_name(dataset, test_key_set, name2id_path=None):
        #self.name2id_path = '/'.join(self.dataset_path.split('/')[:-1])
        #self.dataset_name = self.dataset_path.split('/')[-1].split('.')[0]
        if name2id_path is None:
            name2id_path = '/'.join(dataset.dataset_path.split('/')[:-1])
            dataset_name = dataset.dataset_path.split('/')[-1].split('.')[0]
            name2id_file= name2id_path+'/'+dataset_name+'_name2id.pt'
            if os.path.exists(name2id_file):
                name2id = torch.load(name2id_file)
            else:
                name2id = {}
                for i in tqdm(range(len(dataset)), 'Indexing Dataset'):
                    try:
                        data = dataset[i]
                    except AssertionError as e:
                        print(i, e)
                        continue
                    name = (
                        '/'.join(data.protein_filename.split('/')[-2:]), 
                        '/'.join(data.ligand_filename.split('/')[-2:])
                            )
                    name2id[name] = i
                torch.save(name2id, name2id_file)
        else:
            name2id = torch.load(name2id_file)
        
        train_idx = []
        test_idx = []
        for k,v in tqdm(name2id.items(), 'Spliting'):
            if k in test_key_set:
                test_idx.append(v)
            else:
                train_idx.append(v)
        split_dict = {
            'valid':torch.LongTensor(test_idx), 'train':torch.LongTensor(train_idx)
            }
        subsets = {k:Subset(dataset, indices=v) for k, v in split_dict.items()}
        train_set, val_set = subsets['train'], subsets['valid']
        return train_set, val_set


class CrossDocked2020(object):
    def __init__(self, raw_path, index_path, unexpected_sample=[],
                 atomic_numbers=[6,7,8,9,15,16,17,35,53]):
        self.raw_path = raw_path
        self.file_dirname = os.path.dirname(raw_path)
        self.index_path = index_path
        self.unexpected_sample = unexpected_sample
        self.index = self.get_file(index_path, raw_path)
        self.atomic_numbers = set(atomic_numbers)

    @staticmethod
    def get_file(index_dirname, index_path):
        with open(index_dirname, 'rb') as f:
            index = pickle.load(f)
        file_list = []
        for i in index:
            if i[0] is None:
                continue
            else:
                pdb = os.path.join(index_path, i[0])
            sdf = os.path.join(index_path, i[1])
            file_list.append([pdb,sdf])
        return file_list

    def process(self, raw_file_info):
        try:
            pocket_file, ligand_file = raw_file_info
            lig = Ligand(ligand_file, removeHs=True, sanitize=True)
            for a in lig.mol.GetAtoms():
                if a.GetAtomicNum() not in self.atomic_numbers:
                    return None
            else:
                ligand_dict = lig.to_dict()
            if self.only_backbone:
                pocket_dict = Protein(pocket_file).get_backbone_dict(removeHs=True)
            else:
                pocket_dict = Protein(pocket_file).get_atom_dict()
            #ligand_dict = parse_sdf_to_dict(ligand_fn)
            data = ComplexData.from_protein_ligand_dicts(
                protein_dict=torchify_dict(pocket_dict),
                ligand_dict=torchify_dict(ligand_dict),
            )
            data.protein_filename = '/'.join(pocket_file.split('/')[-2:])
            data.ligand_filename = '/'.join(ligand_file.split('/')[-2:])
            return data
        except:
            return None
    
    def run(self, dataset_name='crossdocked_pocket10_processed.lmdb', lmdb_path=None, 
            max_ligand_atom=50, only_backbone=False, n_process=16, interval=2000):
        #file_dirname = os.path.dirname(self.index_path)
        self.only_backbone = only_backbone
        if lmdb_path:
            lmdb_path = os.path.join(lmdb_path, dataset_name.split('/')[-1])
        else:
            lmdb_path = dataset_name #os.path.join(self.file_dirname,dataset_name.split('/')[-1])
        if os.path.exists(lmdb_path):
            raise FileExistsError(lmdb_path + ' has been existed !')
        db = lmdb.open(
                    lmdb_path,
                    map_size=200*(1024*1024*1024),   # 200GB
                    create=True,
                    subdir=False,
                    readonly=False, # Writable
                )
        data_ix_list = []
        exception_list = []
        data_ix = 0
        for idx in tqdm(range(0, len(self.index), interval)):
            if idx+interval >= len(self.index):
                raw_files = self.index[idx:]
            else:
                raw_files = self.index[idx:idx+interval]
            val_raw_files = []
            for items in raw_files:
                '''if items[0] is None:
                    continue'''
                if 'ATOM' not in open(items[0]).read():
                    continue
                elif items[1] in self.unexpected_sample:
                    continue
                elif Chem.MolFromMolFile(items[1])!=None:
                    val_raw_files.append(items)
                else:
                    exception_list.append(items)
            torch.multiprocessing.set_sharing_strategy('file_system')
            pool = Pool(processes=n_process)
            data_list = pool.map(self.process, val_raw_files)
            with db.begin(write=True, buffers=True) as txn:
                for data in data_list:
                    if data is None: continue
                    if data.protein_pos.size(0) < 50: continue
                    if len(data.ligand_nbh_list) > max_ligand_atom: continue
                    key = str(data_ix).encode()
                    txn.put(
                        key = key,
                        value = pickle.dumps(data)
                    )
                    data_ix_list.append(key)
                    data_ix += 1
        db.close()
        data_ix_list = np.array(data_ix_list)
        np.save(lmdb_path.split('.')[0]+'_Keys', data_ix_list)
        with open(lmdb_path.split('.')[0]+'_invalid.list','w') as fw:
            fw.write(str(exception_list))
