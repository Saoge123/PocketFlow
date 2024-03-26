import torch
from pocket_flow.gdbp_model import PocketFlow
from pocket_flow.utils import Experiment, LoadDataset
from pocket_flow.utils.transform import *
from pocket_flow.utils.data import ComplexData, torchify_dict
import os
from easydict import EasyDict

torch.multiprocessing.set_sharing_strategy('file_system')

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


encoder_cfg = EasyDict(
        {'edge_channels':8, 'num_interactions':6, 'num_heads':4,
     'knn':16, 'cutoff':10.0}
     )
focal_net_cfg = EasyDict(
    {'hidden_dim_sca':64, 'hidden_dim_vec':8}
    )
atom_flow_cfg = EasyDict(
    {'hidden_dim_sca':64, 'hidden_dim_vec':8, 'num_flow_layers':6}
    )
pos_predictor_cfg = EasyDict(
    {'num_filters':[64,64], 'n_component':3}
    )
pos_filter_cfg = EasyDict(
    {'edge_channels':8, 'num_filters':[64,16]}
    )
edge_flow_cfg = EasyDict(
    {'edge_channels':16, 'num_filters':[64,8], 'num_bond_types':3,
     'num_heads':4, 'cutoff':10.0, 'num_flow_layers':6}
    )
config = EasyDict(
    {'deq_coeff':0.9, 'hidden_channels':64, 'hidden_channels_vec':16, 'use_conv1d':False, 'num_bond_types':4,
     'bottleneck':(8,2), 'protein_atom_feature_dim':27, 'ligand_atom_feature_dim':15, 'num_atom_type':9,
     'msg_annealing':True, 'encoder':encoder_cfg, 'atom_flow':atom_flow_cfg, 'pos_predictor':pos_predictor_cfg,
     'edge_flow':edge_flow_cfg, 'focal_net':focal_net_cfg, 'pos_filter':pos_filter_cfg}
    )

protein_featurizer = FeaturizeProteinAtom()
ligand_featurizer = FeaturizeLigandAtom(atomic_numbers=[6,7,8,9,15,16,17,35,53])
traj_fn = LigandTrajectory(perm_type='mix', num_atom_type=9)
focal_masker = FocalMaker(r=6.0, num_work=16, atomic_numbers=[6,7,8,9,15,16,17,35,53])
atom_composer = AtomComposer(
    knn=16, num_workers=16, graph_type='knn', radius=10.0, use_protein_bond=False
    )
combine = Combine(traj_fn, focal_masker, atom_composer, lig_only=True)
transform = TrajCompose([
    RefineData(),
    LigandCountNeighbors(),
    protein_featurizer,
    ligand_featurizer,
    combine,
    collate_fn
])

dataset = LoadDataset('pretrain_data/ZINC_PretrainingDataset.lmdb', transform=transform)
print('Num data:', len(dataset))
train_set, valid_set = LoadDataset.split(dataset, val_num=10000, shuffle=True, random_seed=0)

model = PocketFlow(config).to('cuda:0')
print(model.get_parameter_number())
optimizer = torch.optim.Adam(model.parameters(), lr=2.e-4, weight_decay=0, betas=(0.99, 0.999))
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.6, patience=10, min_lr=1.e-5)

exp = Experiment(
    model, train_set, optimizer, valid_set=valid_set, scheduler=scheduler,
    device='cuda:0', data_parallel=False, use_amp=False
    )
exp.fit_step(
    1000000, valid_per_step=5000, train_batch_size=256, valid_batch_size=512, print_log=False,
    with_tb=True, logdir='./pretraining_log', schedule_key='loss', num_workers=16, 
    pin_memory=False, follow_batch=[], exclude_keys=[], collate_fn=None, 
    max_edge_num_in_batch=2000000
    )
